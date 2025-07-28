# chatbot.py (Final Gemma Version with Stutter Fix)

# --- Core Imports ---
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS
import sounddevice as sd
from scipy.io.wavfile import write, read
from llama_cpp import Llama

# --- VAD, Threading, and Interrupt Imports ---
import collections
import numpy as np
# We no longer need pyaudio
import webrtcvad
import wave
import threading
import queue
import time

# --- Helper Imports ---
import os
import sys
import traceback
from prompts import create_prompt, get_opening_line
import emoji

# --- Suppress TTS Library's Noisy Logging ---
class SuppressTTSOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# --- Configuration ---
MODEL_PATH = r".\gemma-2b-it-Q4_K_M.gguf"
VOICE_TO_CLONE_PATH = r".\my_voice.wav"
WHISPER_MODEL_SIZE = "base.en"
VAD_SILENCE_TIMEOUT = 300
TTS_SPEED = 1.25
AUDIO_FILE = "user_input.wav"
TTS_OUTPUT_FILE = "ai_response.wav"
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_CHUNK_SIZE = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)
TTS_SAMPLE_RATE = 24000

# --- System Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"--- System running on {DEVICE} ---")


# --- VAD Listener Class (Using sounddevice) ---
class VADListener:
    def __init__(self, text_queue, whisper_model, bot_is_speaking_event):
        self.vad = webrtcvad.Vad(3)
        self.text_queue = text_queue
        self.whisper = whisper_model
        self.bot_is_speaking_event = bot_is_speaking_event
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        print("\n--- Continuous Listener Initialized ---")

    def _audio_callback(self, indata, frames, time, status):
        """This is called on a separate thread for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def _transcribe_audio(self, audio_data):
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(audio_np, beam_size=5)
        text = "".join(segment.text for segment in segments).strip()
        if text: self.text_queue.put(text)

    def listen_loop(self):
        print("Listening silently in the background...")
        ring_buffer = collections.deque(maxlen=30)
        triggered = False
        frames = []
        silence_frames = 0

        with sd.RawInputStream(samplerate=VAD_SAMPLE_RATE, blocksize=VAD_CHUNK_SIZE, dtype='int16', channels=1, callback=self._audio_callback):
            while not self.stop_event.is_set():
                if self.bot_is_speaking_event.is_set():
                    time.sleep(0.1)
                    with self.audio_queue.mutex:
                        self.audio_queue.queue.clear()
                    continue
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    is_speech = self.vad.is_speech(data, VAD_SAMPLE_RATE)
                    if not triggered:
                        ring_buffer.append((data, is_speech))
                        if len([f for f, speech in ring_buffer if speech]) > 0.8 * ring_buffer.maxlen:
                            triggered = True; print("Voice detected, recording...")
                            frames.extend([f[0] for f in ring_buffer]); ring_buffer.clear()
                    else:
                        frames.append(data)
                        if not is_speech:
                            silence_frames += 1
                            if silence_frames > (VAD_SILENCE_TIMEOUT // VAD_FRAME_MS):
                                print("End of speech detected.")
                                threading.Thread(target=self._transcribe_audio, args=(b''.join(frames),)).start()
                                triggered, frames, silence_frames = False, [], 0
                                print("\nListening silently in the background...")
                        else:
                            silence_frames = 0
                except queue.Empty:
                    continue

    def stop(self): self.stop_event.set()

# --- Audio Player Class (Using sounddevice) ---
class AudioPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_playing = threading.Event()

    def play_loop(self):
        with sd.RawOutputStream(samplerate=TTS_SAMPLE_RATE, blocksize=1024, dtype='int16', channels=1) as stream:
            while not self.stop_event.is_set():
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    stream.write(audio_chunk)
                    self.is_playing.set()
                except queue.Empty:
                    self.is_playing.clear()

    def add_to_queue(self, audio_data): self.audio_queue.put(audio_data)
    def interrupt(self):
        with self.audio_queue.mutex: self.audio_queue.queue.clear()
        self.is_playing.clear()
    def stop(self): self.stop_event.set()

# --- Main Application Class ---
class Chatbot:
    def __init__(self):
        print("--- Initializing Chatbot ---")
        if not os.path.exists(VOICE_TO_CLONE_PATH): print(f"FATAL ERROR: Voice cloning file not found at '{VOICE_TO_CLONE_PATH}'"); exit()
        self.llm = self.load_llm(); self.tts = self.load_tts()
        self.whisper = self.load_whisper(); self.text_queue = queue.Queue()
        self.audio_player = AudioPlayer(); self.bot_is_speaking = threading.Event()
        self.listener = VADListener(self.text_queue, self.whisper, self.bot_is_speaking)
        print("--- Chatbot Initialized ---")

    def load_llm(self): print(f"Loading brain ({os.path.basename(MODEL_PATH)})..."); return Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    def load_tts(self):
        print("Loading mouth (XTTS v2)...")
        with SuppressTTSOutput():
            return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    def load_whisper(self): print(f"Loading ears (Whisper {WHISPER_MODEL_SIZE})..."); return WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    def think(self, text):
        if not text: return "I didn't hear that."
        print(f"You: {text}")
        prompt = create_prompt(text)
        print("Linie is thinking...")
        output = self.llm(prompt, max_tokens=150, stop=["<start_of_turn>", "<end_of_turn>"], echo=False)
        response = output['choices'][0]['text'].strip()
        cleaned_response = emoji.replace_emoji(response, replace='')
        return cleaned_response

    def speak(self, text):
        if not text: return
        print(f"Linie (Cloned Voice): {text}")
        try:
            with SuppressTTSOutput():
                self.tts.tts_to_file(text=text, file_path=TTS_OUTPUT_FILE, speaker_wav=VOICE_TO_CLONE_PATH, language="en", speed=TTS_SPEED)
            _, audio_data = read(TTS_OUTPUT_FILE)
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                self.audio_player.add_to_queue(audio_data[i:i+chunk_size].tobytes())
        except Exception as e: print(f"Error during speech synthesis: {e}")

    def start_chat(self):
        self.speak_blocking(get_opening_line())
        listener_thread = threading.Thread(target=self.listener.listen_loop, daemon=True)
        player_thread = threading.Thread(target=self.audio_player.play_loop, daemon=True)
        listener_thread.start(); player_thread.start()
        try:
            while True:
                try:
                    user_input = self.text_queue.get(timeout=0.1)
                    if self.audio_player.is_playing.is_set(): print("\nInterrupting Linie..."); self.audio_player.interrupt()
                    if any(x in user_input.lower() for x in ["goodbye", "exit", "shut down"]): self.speak_blocking("Fine. Leaving."); break
                    threading.Thread(target=self._process_and_speak, args=(user_input,)).start()
                except queue.Empty: continue
        except KeyboardInterrupt: print("\nShutdown requested...")
        finally: self.listener.stop(); self.audio_player.stop()

    def _process_and_speak(self, user_input):
        self.bot_is_speaking.set()
        ai_response = self.think(user_input)
        self.speak(ai_response)
        while self.audio_player.is_playing.is_set(): time.sleep(0.1)
        time.sleep(0.4); self.bot_is_speaking.clear()

    def speak_blocking(self, text):
        if not text: return
        self.bot_is_speaking.set()
        print(f"Linie (Cloned Voice): {text}")
        with SuppressTTSOutput():
            self.tts.tts_to_file(text=text, file_path=TTS_OUTPUT_FILE, speaker_wav=VOICE_TO_CLONE_PATH, language="en", speed=TTS_SPEED)
        samplerate, data = read(TTS_OUTPUT_FILE)
        sd.play(data, samplerate); sd.wait()
        time.sleep(0.4); self.bot_is_speaking.clear()

if __name__ == "__main__":
    bot = Chatbot()
    bot.start_chat()

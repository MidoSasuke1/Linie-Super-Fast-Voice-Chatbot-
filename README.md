#                     ### PLEASE READ CAREFULLY 
# 🧊 Linie - Cold Personal Assistant (Voice Chatbot)   "" Yes She is from Frieren Anime ""

Linie is a real-time voice chatbot that listens, thinks, and replies in a **cold, sarcastic** tone using a cloned voice. Powered by local LLMs and TTS/STT models — no API keys needed.

---

## 🧠 Features
- ✅ This model is SUPER fast with minimum dely possible
- ✅ Voice-to-text using **Whisper**
- ✅ Text generation using **Gemma 2B**, Phi, or any compatible `.gguf` model
- ✅ Voice cloning with **XTTS v2**
- ✅ Real-time audio using VAD and `sounddevice`
- ✅ Interruptible TTS if you speak mid-response
- ✅ Runs **entirely offline** with GPU acceleration

---

## 🛠️ Installation Guide : Do it in order to avoid errors and USE POWERSHELL

### 🔹 1. Create the Conda Environment ( Use Powershell) 

```bash
conda create -n chatbot python=3.10 --yes
conda activate chatbot
```

### 🔹 2. Install System Libraries

```bash
conda install -c conda-forge cudnn portaudio --yes
```

### 🔹 3. Set CUDA Build Flags (for llama-cpp-python)

Use **PowerShell** (not CMD):

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
```

### 🔹 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 5. Install PyTorch (GPU Version)

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## 📁 Required Files

Place these in the **same folder as `chatbot.py`**:

- ✅ `gemma-2b-it-Q4_K_M.gguf` or another `.gguf` model
- ✅ `my_voice.wav` — your cloned voice sample (30–60s recommended)



## 🚀 Run the Bot

```bash
python chatbot.py
```

#               Finished 

-----------------------------------------------------------------

---
##                Things you can change to your suit::

## 🗣️ Whisper Model Options

In `chatbot.py`, you can change whisper model by tweaking the following line of code. "base.en" model is recommended for balance quality and speed,
"Large-v3" is slower but has a very high quality to capture voice, "tiny.en" is super fast but low quality.

#The line of code to tweak:
```python
WHISPER_MODEL_SIZE = "base.en"
```

#The available models for quality and language:

**English-only models**: recommended us 'base.en' for fast response. You can use 'tiny.en' for fastest response but it has some artifacts as mentioned.
- `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3`, `large-v2`

**Multilingual** you can use these models but the ai will response in english only. This is good for translation.
The code need a very little of modification to response in Japanese but that's for another day:
- `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v2`

---



---

## 👤 Linie’s Personality (Here you can manipulate HER personality):

Linie is sarcastic, cold, and unimpressed. You can modify this in `prompts.py`:

```python
"You are Linie, a personal assistant. You are mean, cold, and unimpressed. NEVER mention you are an AI..."
```

---

## 💡 Model Switching

You can swap the model used by editing this line in `chatbot.py`:

```python
MODEL_PATH = "./gemma-2b-it-Q4_K_M.gguf"
```

Compatible `.gguf` models include:

- `gemma-2b-it-Q4_K_M.gguf` (best for following prompts and good speed)  "Recommended"
- `phi-2-Q4_K_M.gguf` (faster but less smart)  
- `Phi-3-mini-4k-instruct-q4.gguf` (slower but smarter a little)

---



##Important       Things to keep in mind

- Since the goal was to shrink the delay time there're few downsides. The AI is always listening so when you start speaking and
  it recognize your speech it will start proceeding, the moment you stop talking there's a delay time in which your speech will
  be sent to the LLM to be processed (the Time is 300 milliseconds), this consider a very short time by stander. So you need to eighter not
  to pause while you speaking or increase this time to a higher value like (1000 = 1 second) or (3000 = 3 seconds), so the ai will wait 3 seconds
  or so before grap that chunk of audio.

  Edit this file:
  
```bash
python chatbot.py
```

The Line of Code to change:
```
VAD_SILENCE_TIMEOUT = 300  #the time the ai will wait after you stop talking before it sends your speech to LLM
```


## Note about the AI Voice:
- When She (AI) starts speaking you will notice a small cut in the sound at the beginning (Fracture of second), and that's normal
  since the voice is being sent as a stream of chunks, in other words instead of generating the whole audio then we play it;
  we generate the first one or two words and play it immediatly while the rest of the audio is being generated.
 It's onlly at the start, could get annoying but it's a speed/quality trade-off.

You can Replace this file "Chatbot.py" by this "chatbot Backup.py" which is using different method for clean voice but
the generation will be (0.5 - 1.5) seconds longer.


## 📦 Credits

- [Coqui TTS (XTTS v2)](https://github.com/coqui-ai/TTS)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Hugging Face LLMs](https://huggingface.co)

---

## ⚠️ Disclaimer
Voice cloning is for educational and personal use. Please do not use any voices you don’t have permission to use.


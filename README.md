#                     ### PLEASE READ CAREFULLY 
# 🧊 Linie - Cold Personal Assistant (Voice Chatbot)

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

---

## 🗣️ Whisper Model Options

In `chatbot.py`, you can change:

```python
WHISPER_MODEL_SIZE = "base.en"
```

**English-only models**: recommended us 'base.en' for fast response. You can use 'tiny.en' for fastest response but it has some artifacts.
- `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3`, `large-v2`

**Multilingual** you can use these models but the ai will response in english only. the code need a very little of modification to response in Japanese
but that for the future:
- `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v2`

---

## 🚀 Run the Bot

```bash
python chatbot.py
```

---

## 👤 Linie’s Personality

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

- `gemma-2b-it-Q4_K_M.gguf` (best for following prompts)
- `phi-2-Q4_K_M.gguf` (faster but less smart)
- `Phi-3-mini-4k-instruct-q4.gguf` (slower but smarter)

---

## 📦 Credits

- [Coqui TTS (XTTS v2)](https://github.com/coqui-ai/TTS)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Hugging Face LLMs](https://huggingface.co)

---

## ⚠️ Disclaimer
Voice cloning is for educational and personal use. Please do not use any voices you don’t have permission to use.


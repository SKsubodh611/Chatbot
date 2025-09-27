#    working fine
#     the content plays after first is complete 

import os
import io
import time
import wave
import numpy as np
import winsound 
import multiprocessing as mp
import sounddevice as sd
from openai import OpenAI

# ---------------- CONFIG ----------------
MODEL_STT = "whisper-1"
MODEL_CHAT = "gpt-4o-mini"   # lightweight chat model
MODEL_TTS = "tts-1"
TTS_VOICE = "alloy"
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# ---------------- UTIL ----------------
def wav_bytes_to_rms(frames: bytes) -> float:
    """RMS of int16 PCM bytes"""
    if not frames:
        return 0.0
    count = len(frames) // SAMPLE_WIDTH
    ints = np.frombuffer(frames, dtype=np.int16)
    return float(np.sqrt(np.mean(ints.astype(np.float32) ** 2)) + 1e-9)

def frames_to_wav_io(frames: list[bytes], sr=SAMPLE_RATE) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sr)
        wf.writeframes(b"".join(frames))
    buf.seek(0)
    return buf

# ---------------- LISTENER ----------------
def proc_listener(q_audio_frames: mp.Queue, stop_event: mp.Event):
    print("[Listener] starting...")
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS
    block_size = int(SAMPLE_RATE * 0.02)  # 20ms blocks

    # Calibrate noise floor
    print("[Listener] calibrating...")
    cal_blocks = int(1.0 * SAMPLE_RATE // block_size)
    noise_vals = []
    with sd.InputStream(dtype="int16", blocksize=block_size) as stream:
        for _ in range(cal_blocks):
            block, _ = stream.read(block_size)
            noise_vals.append(wav_bytes_to_rms(block.tobytes()))
            time.sleep(0.02)
    noise_rms = np.median(noise_vals) if noise_vals else 200
    vad_threshold = max(150.0, noise_rms * 1.5)
    print(f"[Listener] threshold={vad_threshold:.1f}")

    speaking = False
    last_voice_time = 0.0

    with sd.InputStream(dtype="int16", blocksize=block_size) as stream:
        print("[Listener] ready. Speak now!")
        while True:
            block, _ = stream.read(block_size)
            block_bytes = block.tobytes()
            energy = wav_bytes_to_rms(block_bytes)
            now = time.time()

            if energy >= vad_threshold:
                if not speaking:
                    speaking = True
                    stop_event.set()
                q_audio_frames.put(block_bytes)
                last_voice_time = now
            else:
                if speaking:
                    q_audio_frames.put(block_bytes)

            # end utterance if silence too long
            if speaking and (now - last_voice_time) > 0.7:
                speaking = False
                q_audio_frames.put(None)
                print("[Listener] utterance ready")

# ---------------- TRANSCRIBER ----------------
def proc_transcriber(q_audio_frames: mp.Queue, q_text: mp.Queue, api_key: str):
    print("[Transcriber] starting...")
    client = OpenAI(api_key=api_key)
    frames: list[bytes] = []

    while True:
        chunk = q_audio_frames.get()
        if chunk is None:
            if not frames:
                continue
            wav_buf = frames_to_wav_io(frames)
            buf = io.BytesIO(wav_buf.getvalue())
            buf.name = "utt.wav"

            try:
                tr = client.audio.transcriptions.create(
                    model=MODEL_STT,
                    file=buf,
                    language="en"  # force English; change to "hi" for Hindi
                )
                text = (tr.text or "").strip()
                if text:
                    print(f"[Transcriber] {text}")
                    q_text.put(text)
            except Exception as e:
                print(f"[Transcriber error] {e}")
            frames.clear()
        else:
            frames.append(chunk)

# ---------------- BRAIN ----------------
def proc_brain(q_text: mp.Queue, q_tts: mp.Queue, stop_event: mp.Event, api_key: str):
    print("[Brain] starting...")
    client = OpenAI(api_key=api_key)
    history = [{"role": "system", "content": "You are a helpful conversational voice assistant."}]
    while True:
        user_text = q_text.get()
        if not user_text:
            continue
        stop_event.set()
        history.append({"role": "user", "content": user_text})
        try:
            resp = client.chat.completions.create(model=MODEL_CHAT, messages=history)
            reply = resp.choices[0].message.content
            history.append({"role": "assistant", "content": reply})
            print(f"AI: {reply}")
            q_tts.put(reply)
        except Exception as e:
            print(f"[Brain error] {e}")

# ---------------- SPEAKER ----------------
def normalize_wav_bytes(wav_bytes: bytes) -> bytes:
    """Ensure WAV is 16-bit PCM (winsound-compatible)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as src:
        channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        frames = src.readframes(src.getnframes())

    if sampwidth != 2:  # convert to 16-bit
        arr = np.frombuffer(frames, dtype=np.float32)
        arr16 = (arr * 32767).astype(np.int16)
        frames = arr16.tobytes()
        sampwidth = 2

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(frames)
    return out.getvalue()

def tts_wav_bytes(client: OpenAI, text: str) -> bytes:
    resp = client.audio.speech.create(
        model=MODEL_TTS,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    wav_bytes = (
        bytes(resp.content)
        if hasattr(resp, "content") and isinstance(resp.content, (bytes, bytearray))
        else resp.read() if hasattr(resp, "read") else bytes(resp)
    )
    return normalize_wav_bytes(wav_bytes)

def proc_speaker(q_tts: mp.Queue, stop_event: mp.Event, api_key: str):
    print("[Speaker] starting...")
    import tempfile
    client = OpenAI(api_key=api_key)
    while True:
        text = q_tts.get()
        if not text:
            continue
        if stop_event.is_set():
            stop_event.clear()

        try:
            wav_data = tts_wav_bytes(client, text)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(wav_data)
            tmp.close()
            winsound.PlaySound(tmp.name, winsound.SND_FILENAME)
        except Exception as e:
            print(f"[Speaker error] {e}")

# ---------------- MAIN ----------------
def main():
    mp.set_start_method("spawn", force=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        print("❌ Set your API key:  setx OPENAI_API_KEY \"sk-...\"  (then reopen terminal)")
        return

    manager = mp.Manager()
    q_audio_frames = manager.Queue()
    q_text = manager.Queue()
    q_tts = manager.Queue()
    stop_event = manager.Event()

    listener = mp.Process(target=proc_listener,   args=(q_audio_frames, stop_event))
    transcr  = mp.Process(target=proc_transcriber,args=(q_audio_frames, q_text, api_key))
    brain    = mp.Process(target=proc_brain,     args=(q_text, q_tts, stop_event, api_key))
    speaker  = mp.Process(target=proc_speaker,   args=(q_tts, stop_event, api_key))

    listener.start()
    transcr.start()
    brain.start()
    speaker.start()

    print("🎤 Voice AI running. Press Ctrl+C to stop.")
    try:
        listener.join()
        transcr.join()
        brain.join()
        speaker.join()
    except KeyboardInterrupt:
        print("\n[Main] stopping...")
        for p in (listener, transcr, brain, speaker):
            if p.is_alive():
                p.terminate()

if __name__ == "__main__":
    main()

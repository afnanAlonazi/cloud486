import psutil
import time
import whisper

# Load model
model = whisper.load_model("tiny")

# Load audio file
audio_path = "AudioWAV/1091_ITS_ANG_XX.wav"

# Track CPU & RAM usage
cpu_usage_before = psutil.cpu_percent()
memory_before = psutil.virtual_memory().percent

start_time = time.time()
print("Transcribing audio using Whisper...")
result = model.transcribe(audio_path)
end_time = time.time()

cpu_usage_after = psutil.cpu_percent()
memory_after = psutil.virtual_memory().percent

# Print results
transcription = result["text"]
print("Transcription:", transcription)
print(f"🕒 Inference Time: {end_time - start_time:.2f} seconds")
print(f"🔥 CPU Usage Before: {cpu_usage_before}% After: {cpu_usage_after}%")
print(f"💾 Memory Usage Before: {memory_before}% After: {memory_after}%")

# Save transcription
with open("transcription.txt", "w") as file:
    file.write(transcription)
print("✅ Transcription has been saved to 'transcription.txt'.")

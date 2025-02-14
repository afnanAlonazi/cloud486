# import psutil
# import time
# import whisper

# # Load model
# model = whisper.load_model("tiny")

# # Load audio file
# audio_path = "AudioWAV/1091_ITS_ANG_XX.wav"

# # Track CPU & RAM usage
# cpu_usage_before = psutil.cpu_percent()
# memory_before = psutil.virtual_memory().percent

# start_time = time.time()
# print("Transcribing audio using Whisper...")
# result = model.transcribe(audio_path)
# end_time = time.time()

# cpu_usage_after = psutil.cpu_percent()
# memory_after = psutil.virtual_memory().percent

# # Print results
# transcription = result["text"]
# print("Transcription:", transcription)
# print(f"🕒 Inference Time: {end_time - start_time:.2f} seconds")
# print(f"🔥 CPU Usage Before: {cpu_usage_before}% After: {cpu_usage_after}%")
# print(f"💾 Memory Usage Before: {memory_before}% After: {memory_after}%")

# # Save transcription
# with open("transcription.txt", "w") as file:
#     file.write(transcription)
# print("✅ Transcription has been saved to 'transcription.txt'.")
import whisper
import time
import psutil

# Define models and audio files
models = ["tiny", "small", "base", "medium"]
audio_files = {
    "2 sec": "AudioWAV/2sec.wav", 
    "10 sec": "AudioWAV/10sec.wav", 
    "20 sec": "AudioWAV/20sec.wav", 
    "30 sec": "AudioWAV/30sec.wav", 
    "34 sec": "AudioWAV/34sec.wav", 
    "45 sec": "AudioWAV/45sec.wav", 
    "1 min": "AudioWAV/60sec.wav",
    "2min ": "AudioWAV/LDC2004S13.wav"
    
}

# Open file for writing transcriptions and benchmark results
with open("transcription.txt", "w", encoding="utf-8") as file:
    # Run benchmarking
    for model_name in models:
        print(f"\n🔹 Benchmarking Whisper Model: {model_name.upper()}")
        file.write(f"\n🔹 Benchmarking Whisper Model: {model_name.upper()}\n")

        # Load Whisper model
        model = whisper.load_model(model_name)

        for length, audio_path in audio_files.items():
            print(f"\n🎙️ Testing with Audio Length: {length}")
            file.write(f"\n🎙️ Audio Length: {length}\n")

            # Measure CPU & RAM before transcription
            cpu_before = psutil.cpu_percent()
            ram_before = psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB

            # Start timing
            start_time = time.time()
            result = model.transcribe(audio_path)
            end_time = time.time()

            # Measure CPU & RAM after transcription
            cpu_after = psutil.cpu_percent()
            ram_after = psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB

            # Get full transcription
            transcription = result["text"]

            # Print results
            print(f"🕒 Inference Time: {round(end_time - start_time, 2)} seconds")
            print(f"🔥 CPU Usage Before: {cpu_before}%, After: {cpu_after}%")
            print(f"💾 Memory Usage Before: {round(ram_before, 2)} MB, After: {round(ram_after, 2)} MB")
            print(f"📜 Transcription Snippet: {transcription[:100]}...")  # Show first 100 characters of transcription

            # Write results to file
            file.write(f"🕒 Inference Time: {round(end_time - start_time, 2)} seconds\n")
            file.write(f"🔥 CPU Usage Before: {cpu_before}%, After: {cpu_after}%\n")
            file.write(f"💾 Memory Usage Before: {round(ram_before, 2)} MB, After: {round(ram_after, 2)} MB\n")
            file.write(f"📜 Full Transcription:\n{transcription}\n")
            file.write("=" * 50 + "\n")  # Separator

print("\n✅ Benchmarking Complete! All results saved in 'transcription.txt'.")

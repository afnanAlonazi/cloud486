
import whisper
import time
import psutil

# Define models and audio files
models = ["tiny", "small", "base", "medium"]
audio_files = {
    "2 sec": "AudioWAV/2sec.wav", # source: https://github.com/jim-schwoebel/voice_datasets
    "10 sec": "AudioWAV/10sec.wav", # source: https://vt.tiktok.com/ZSMrQPrgf/
    "20 sec": "AudioWAV/20sec.wav", # source: https://vt.tiktok.com/ZSMrQGXTS/
    "30 sec": "AudioWAV/30sec.wav", # source: https://vt.tiktok.com/ZSMrQTuMR/
    "34 sec": "AudioWAV/34sec.wav", # source: https://vt.tiktok.com/ZSMrQ3dqQ/
    "45 sec": "AudioWAV/45sec.wav", # source: https://vt.tiktok.com/ZSMrC8Avb/
    "1 min": "AudioWAV/60sec.wav",  # source:  https://vt.tiktok.com/ZSMrQKK51/
    "2min ": "AudioWAV/LDC2004S13.wav" # source: https://catalog.ldc.upenn.edu/LDC2004S13
    
}

# Open file for writing transcriptions and benchmark results
with open("results.txt", "w", encoding="utf-8") as file:
    # Run benchmarking
    for model_name in models:
        print(f"\n🔹 Benchmarking Whisper Model: {model_name.upper()}")
        file.write(f"\n🔹 Benchmarking Whisper Model: {model_name.upper()}\n")

        # Load Whisper model
        model = whisper.load_model(model_name)

        for length, audio_path in audio_files.items():
            print(f"\n🎙️ Testing with Audio Length: {length}")
            file.write(f"\n🎙️ Audio Length: {length}\n")

            # Measure CPU & RAM before processing
            cpu_before = psutil.cpu_percent()
            ram_before = psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB

            # Start timing for response time
            response_start = time.time()

            # Load audio file (preprocessing time)
            audio = whisper.load_audio(audio_path)

            # Start timing for inference time
            inference_start = time.time()
            result = model.transcribe(audio_path)
            inference_end = time.time()

            # End timing for response time
            response_end = time.time()

            # Measure CPU & RAM after processing
            cpu_after = psutil.cpu_percent()
            ram_after = psutil.virtual_memory().used / (1024 * 1024)  # Convert bytes to MB

            # Get full transcription
            transcription = result["text"]

            # Calculate times
            inference_time = round(inference_end - inference_start, 2)
            response_time = round(response_end - response_start, 2)

            # Print results
            print(f"🕒 Inference Time: {inference_time} seconds")
            print(f"⚡ Response Time: {response_time} seconds")
            print(f"🔥 CPU Usage Before: {cpu_before}%, After: {cpu_after}%")
            print(f"💾 Memory Usage Before: {round(ram_before, 2)} MB, After: {round(ram_after, 2)} MB")
            print(f"📜 Transcription Snippet: {transcription[:100]}...")

            # Write results to file
            file.write(f"🕒 Inference Time: {inference_time} seconds\n")
            file.write(f"⚡ Response Time: {response_time} seconds\n")
            file.write(f"🔥 CPU Usage Before: {cpu_before}%, After: {cpu_after}%\n")
            file.write(f"💾 Memory Usage Before: {round(ram_before, 2)} MB, After: {round(ram_after, 2)} MB\n")
            file.write(f"📜 Full Transcription:\n{transcription}\n")
            file.write("=" * 50 + "\n")  # Separator

print("\n✅ Benchmarking Complete! All results saved in 'results.txt'.")
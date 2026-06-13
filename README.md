## About the Project

This project provides a comprehensive comparative analysis evaluating the performance, efficiency, and resource utilization of AI-driven Speech Recognition services deployed locally versus in the cloud. The study is referenced verbatim from the document "report.pdf".

### 🛠️ Technologies & Implementations

The system is fully implemented in **Python** and evaluates two main deployment strategies:
1. **Local Deployment (OpenAI Whisper):** Utilizing the open-source `whisper` library across various model sizes (`tiny`, `base`, `small`, and `medium`) to process audio file transcriptions locally[cite: 1]. Performance, CPU, and memory utilization were monitored using the `psutil` and `time` libraries.
2. **Cloud Deployment (Microsoft Azure Speech-to-Text API):** Utilizing the `azure-cognitiveservices-speech` SDK to send audio data to the cloud, incorporating network latency measurements via the `requests` library and audio processing via `librosa`.

### 📊 Performance Metrics Evaluated
* **Inference & Response Time:** Analyzed across multiple audio lengths ranging from 2 seconds up to 2 minutes.
* **Hardware Resource & Energy Consumption:** Detailed monitoring of CPU, Memory (RAM), GPU utilization, and hardware power fluctuations (Watts and Temperature) during local processing.
* **Network Latency:** Evaluation of cloud network overhead impacts on real-time transcription[cite: 1].
* **Cost & Scalability Analysis:** Financial and architectural trade-offs between local computing requirements and cloud consumption models.

### 📈 Key Findings
* While cloud deployment (**Azure STT**) excels in processing short audio clips with zero local hardware load, local deployment (**Whisper**) becomes remarkably more competitive and highly efficient as the audio length increases. 
* For instance, at the 2-minute mark, `Whisper-Small` achieved a faster response time than Azure STT by bypassing continuous network communication constraints.

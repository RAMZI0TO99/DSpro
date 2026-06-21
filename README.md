# DSpro VRAG (Video RAG)
Agentic Spatio-Temporal RAG API with Tri-Modal Video Search.

This project is a powerful, fully-local AI Video Intelligence platform. It automatically ingests video files, transcribes audio, performs scene detection, reads on-screen text (OCR), and allows you to chat with or semantically search across your video library using Local LLMs.

## Features
- **Tri-Modal Vector Search:** Combines Visual (CLIP), Audio/Text (BM25 + WhisperX), and OCR (EasyOCR) into a unified Qdrant vector database.
- **Dynamic RAG:** Intelligently packs video segments into the LLM context window based on relevance, preventing context overflows.
- **Local First:** Designed to run 100% locally via LM Studio/Ollama, ensuring complete privacy.
- **Hardware Accelerated:** Fully optimized Docker setup with NVIDIA GPU passthrough.

## Quick Start (Docker)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RAMZI0TO99/DSpro_VRAG.git
   cd DSpro_VRAG
   ```

2. **Configure Environment:**
   *(Optional)* Copy the example environment file if you plan to use external API keys (OpenAI/Gemini).
   ```bash
   cp .env.example .env
   ```

3. **Boot the platform:**
   This will download all necessary Machine Learning models (Whisper, CLIP, EasyOCR) into persistent volumes and launch the FastAPI backend and Qdrant database.
   ```bash
   docker-compose up --build
   ```

4. **Connect your Local LLM:**
   By default, the UI is configured to talk to a local LM Studio instance running on your host machine.
   - Start LM Studio on your host machine and start the local server on port `1234`.
   - Open your browser to `http://localhost:8000/` to access the UI.
   - The UI is pre-configured to communicate with LM Studio via Docker's internal routing (`host.docker.internal`).

## Architecture
- **Frontend:** Vanilla HTML/CSS/JS (Served dynamically by FastAPI)
- **Backend:** FastAPI (Python)
- **Database:** Qdrant (Reciprocal Rank Fusion hybrid search)
- **ML Pipeline:** PyTorch, WhisperX, OpenCLIP, EasyOCR, PySceneDetect

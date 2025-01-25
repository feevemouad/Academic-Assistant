# Academic Assistant 🤖

An AI-powered academic assistant with document management and conversational capabilities using Ollama, MinIO, and Streamlit.

## Features ✨

- **Document Storage**: Secure PDF storage with MinIO object storage
- **AI Conversations**: LLM-powered Q&A with context-aware responses
- **Vector Search**: Semantic document retrieval using Ollama embeddings
- **Event-Driven Updates**: Automatic index updates via MinIO webhooks
- **Multi-Model Support**: Flexible integration with various LLM providers

## Prerequisites 📋

- Python 3.9+
- [Ollama](https://ollama.ai/) running locally
- MinIO server (included in repository)

## Installation 🛠️

```bash
git clone https://github.com/feevemouad/Academic-Assistant.git
cd Academic-Assistant
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Configuration ⚙️

1. **MinIO Setup**:
   - Create bucket `documents` in MinIO console (http://localhost:9001)
   - Configure event notifications:
     - ARN: `http://localhost:8000/webhook`
     - Events: `Put` and `Delete`

2. **Ollama Setup**:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1
   ```

3. **Environment Setup**:
   Update `config.yml` if needed (defaults work for local setup):
   ```yaml
   minio:
     endpoint: 'localhost:9000'
     access_key: 'minioadmin'
     secret_key: 'minioadmin'
   ```

## Usage 🚀

1. **Start MinIO Server**:
   ```bash
   ./minio.exe server data --console-address ':9001'
   ```

2. **Run Backend API**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level info
   ```

3. **Launch Frontend**:
   ```bash
   cd frontend
   streamlit run main.py --server.headless true
   ```

Access the interface at `http://localhost:8501`

## Project Structure 📁

```
Academic-Assistant/
├── data/               # MinIO storage directory
├── frontend/           # Streamlit UI
│   ├── Utils/          # UI components
│   └── .streamlit/     # Streamlit config
├── src/                # Core logic
│   ├── models/         # Data models
│   ├── vector_store.py # Document processing
│   └── chat_model.py   # Conversation logic
├── config.yml          # Configuration
├── requirements.txt    # Dependencies
└── README.md
```

## API Endpoints 🌐

- `POST /chat` - Process user queries
- `POST /webhook` - Handle MinIO events (auto-configures vector store)

## Troubleshooting 🔧

**Common Issues:**
- Port conflicts: Ensure ports 8000 (API), 8501 (UI), 9000/9001 (MinIO) are free
- MinIO errors: Delete `data/.minio.sys` and restart MinIO
- Model issues: Verify Ollama models are downloaded (`ollama list`)

**Log Locations:**
- API Server: Console output
- MinIO: Console output
- Streamlit: Console output

## Technologies Used 🛠️

- **Ollama** - Local LLM & embeddings
- **MinIO** - Object storage
- **LangChain** - LLM orchestration
- **FAISS** - Vector search
- **Streamlit** - Web interface
- **FastAPI** - REST backend

---

**Note:** For production use:
- Secure MinIO with proper credentials
- Enable HTTPS
- Use persistent storage for MinIO data
- Consider adding authentication layers

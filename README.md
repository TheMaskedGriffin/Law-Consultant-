# Law Consultant: Bharatiya Nyaya Sanhita Legal Assistant

A Retrieval-Augmented Generation (RAG) application that provides intelligent question-answering capabilities for the Bharatiya Nyaya Sanhita (BNS) 2023 legal documents. Built with LangChain, OpenAI, and Streamlit, this assistant leverages vector embeddings, semantic caching, and reranking for accurate and efficient legal information retrieval.

[Click here to access in cloud](https://xktc6xikyy2uj39cjnjhtz.streamlit.app/)

## Features

- **RAG-Based Question Answering**: Retrieves relevant legal context from BNS 2023 documents to provide accurate answers
- **Semantic Caching with Redis**: Implements intelligent caching using cosine similarity to avoid redundant database queries
- **FlashRank Reranking**: Enhances retrieval quality by reranking search results for better relevance
- **Conversational Memory**: Maintains chat history for context-aware follow-up questions
- **Multiple Model Support**: Choose from GPT-4o, GPT-4o-mini, GPT-5, and GPT-5-mini
- **Interactive UI**: User-friendly Streamlit interface for easy interaction

## Architecture

```
User Query → Redis Cache Check → Vector Database (ChromaDB) 
    ↓                ↓
Cache Hit         Cache Miss
    ↓                ↓
Return Answer    Retrieval + Reranking → LLM → Answer → Cache Result
```

## Technology Stack

- **LangChain**: Framework for building LLM applications
- **OpenAI**: GPT models and embeddings (text-embedding-3-small/large)
- **ChromaDB**: Vector database for document storage and retrieval
- **Redis**: Semantic caching layer
- **FlashRank**: Result reranking for improved relevance
- **Streamlit**: Web application framework
- **PyPDF**: PDF document processing

## Prerequisites

- Python 3.8+
- OpenAI API key
- Redis instance (configured via Streamlit secrets)
- BNS 2023 PDF document

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nyayagpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Streamlit secrets for Redis:

Create `.streamlit/secrets.toml`:
```toml
REDIS_HOST = "your-redis-host"
REDIS_PASSWORD = "your-redis-password"
```

4. Add your BNS 2023 PDF:

Place the `BNS_2023.pdf` file in the `data/` directory.

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. In the sidebar:
   - Enter your OpenAI API key
   - Select your preferred model
   - Click "Create Vector Database" (first-time setup only)

3. Enter your legal query in the text input field

4. Receive AI-generated answers based on BNS 2023 legal documents

## Project Structure

```
.
├── main.py              # Main Streamlit application
├── build_db.py          # Vector database creation script
├── redis_client.py      # Redis caching logic with semantic search
├── sqlite_fix.py        # SQLite compatibility fix for ChromaDB
├── requirements.txt     # Python dependencies
├── data/
│   └── BNS_2023.pdf    # Legal document (to be added)
└── db/
    └── bns_db/         # ChromaDB vector store (auto-generated)
```

## Key Components

### Vector Database Creation (`build_db.py`)
- Loads PDF documents using PyPDFLoader
- Splits text into chunks (2000 characters, 500 overlap)
- Creates embeddings using OpenAI's text-embedding-3-small
- Persists vectors in ChromaDB

### Redis Semantic Cache (`redis_client.py`)
- Normalizes and embeds user queries
- Compares query embeddings using cosine similarity (threshold: 0.60)
- Returns cached answers for similar queries
- TTL: 300 seconds (5 minutes)

### Main Application (`main.py`)
- Implements conversational RAG chain with history-aware retrieval
- Uses FlashRank for result reranking
- Maintains chat history for context
- Integrates Redis caching layer

## Configuration

### Retrieval Settings
- **Search Type**: Similarity search
- **Results (k)**: 10 documents
- **Chunk Size**: 2000 characters
- **Chunk Overlap**: 500 characters

### Cache Settings
- **Similarity Threshold**: 0.60
- **TTL**: 300 seconds
- **Embedding Model**: text-embedding-3-large

### Model Options
- gpt-4o
- gpt-4o-mini
- gpt-5
- gpt-5-mini

## Performance Optimization

1. **Semantic Caching**: Reduces API calls by matching similar queries
2. **Result Reranking**: Improves answer quality with FlashRank
3. **Vector Search**: Fast similarity search using ChromaDB
4. **Chunking Strategy**: Optimized chunk size with overlap for context preservation

## Troubleshooting

**Issue**: ChromaDB SQLite errors on some platforms
- **Solution**: The `sqlite_fix.py` module replaces the default SQLite with pysqlite3-binary

**Issue**: Redis connection errors
- **Solution**: Verify Redis credentials in `.streamlit/secrets.toml`

**Issue**: Vector database not found
- **Solution**: Click "Create Vector Database" button in the sidebar

## Future Enhancements

- [ ] Multi-document support (IPC, CrPC, etc.)
- [ ] Citation tracking with source attribution
- [ ] Advanced filtering (sections, chapters)
- [ ] User authentication and query history
- [ ] Export conversation transcripts
- [ ] Fine-tuned embeddings for legal domain

## License

[Add your license information here]

## Acknowledgments

- Bharatiya Nyaya Sanhita 2023 legal documents
- LangChain community for RAG frameworks
- OpenAI for language models and embeddings

## Contact

[Add your contact information here]

---

**Note**: This application is for informational purposes only and should not be considered as legal advice. Always consult with qualified legal professionals for legal matters.

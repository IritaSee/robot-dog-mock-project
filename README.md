# Technical Test-Vidavox

## Overview

This project provides tools for:

- Processing and storing document data in a vector database (Qdrant)
- Retrieving relevant information based on natural language queries
- Generating responses using LLM (Groq) with the retrieved context
- Converting natural language queries to SQL for user-to-robot interaction analysis
- Executing SQL queries against an SQLite of interaction database
- Preparing instruction-response datasets for LLM fine-tuning from PDF documents

## Requirements

- Python 3.12+
- Docker and Docker Compose (for running Qdrant & Postgres)
- Groq API key
- SQLite

## Installation

### 1. Unzip the repository

### 2. Set up Python environment

Using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) poppler install
some might experience `pdf2image.exceptions.PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?`, fix by installing poppler, in MacOS:
```bash
brew install poppler
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
GROQ_API_KEY=your_groq_api_key
GROQ_LLM_NAME=llama3-70b-8192  # or another model supported by Groq
FASTEMBED_MODEL_NAME=mixedbread-ai/mxbai-embed-large-v1
FASTEMBED_VECTOR_DIM=1024
SQLITE_DB_PATH=data/robot_dog_interaction.sqlite  # Path to your earthquake SQLite database
QDRANT_COLLECTION_NAME=robot_dog_collection  # Qdrant collection name
```

If there is any changes to the `.env` file, reload the terminal to load the new value.

## Docker Setup

The project uses Qdrant as a vector database, which is configured in the `docker-compose.yaml` file.

Start the Qdrant service:

```bash
docker compose up -d
```

This will start a Qdrant instance accessible at:

- API: http://localhost:6333
- Web UI: http://localhost:6334

## Usage

### 1. Store documents in the vector database

The `store_data_to_vecdb.py` script processes PDF documents and stores them in the Qdrant vector database:

```bash
python command/store_data_to_vecdb.py
```

By default, it processes PDF documents from the `data/` directory.

### 2. Test retrieval and query the system

The `test_retrieval.py` script demonstrates how to retrieve information from the vector database and generate responses using Groq:

```bash
python command/test_retrieval.py
```

You can modify the query in the script to ask different questions about the mock product.

### 3. Use the LangGraph agent with earthquake tools

The `example_usage.py` script demonstrates how to use the LangGraph agent with the earthquake tools:

```bash
python command/example_usage.py
```

This script initializes the LangGraph agent with the following tools:

- `get_current_timestamp`: Returns the current timestamp
- `retrieve_earthquake_knowledge`: Retrieves earthquake information using RAG
- `query_earthquake_database`: Converts natural language to SQL and queries the earthquake database

### 4. Prepare fine-tuning datasets from PDF documents

The `prepare_finetune_dataset.py` script extracts text from PDFs and generates instruction-response pairs suitable for fine-tuning LLMs:

```bash
python command/prepare_finetune_dataset.py
```

This script supports various options:
```bash
python command/prepare_finetune_dataset.py --llm_model llama-3.3-70b-versatile --num_pairs_per_chunk 5 --format jsonl
```

Key features:
- Extracts text from PDFs using customizable chunking strategies
- Generates high-quality instruction-response pairs using Groq or Ollama LLMs
- Formats datasets in JSONL, CSV, or Parquet with train/validation splits
- Includes document source tracking and metadata

## Available Tools

### RAG Tools

- `retrieve_earthquake_knowledge`: Retrieves earthquake information from the vector database based on natural language queries

### Text-to-SQL Tools

- `query_earthquake_database`: Converts natural language queries to SQL and executes them against the earthquake SQLite database

### Fine-tuning Tools

- `prepare_finetune_dataset.py`: Creates instruction-response datasets from PDF documents for fine-tuning LLMs

## Customization

### Using different documents

To use your own documents:

1. Place your PDF files in the `data/` directory
2. Modify the `document_paths` list in `store_data_to_vecdb.py`
3. Run the script to process and store the documents

### Changing vector database settings

You can modify the following parameters in `store_data_to_vecdb.py`:

- `collection_name`: Name of the Qdrant collection
- `chunking_strategy`: Strategy for chunking documents (e.g., "by_title", "basic")
- `strategy`: Processing strategy (e.g., "hi_res", "fast")

### Using a different embedding model

Update the `FASTEMBED_MODEL_NAME` and `FASTEMBED_VECTOR_DIM` in your `.env` file to use a different embedding model.

### Using a different SQLite database

Update the `SQLITE_DB_PATH` in your `.env` file to point to a different SQLite database file.

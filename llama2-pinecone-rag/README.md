
# Llama2 Pinecone RAG

Retrieval-Augmented Generation project using OpenAI embeddings and Pinecone.
Indexes Llama 2 arXiv papers and performs semantic search over chunks.

## Setup
1. Create a `.env` file with:
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key

2. Install dependencies:
pip install -r requirements.txt

3. Run:
python src/main.py

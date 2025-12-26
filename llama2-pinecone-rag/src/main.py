import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found")

# LLM
chat = ChatOpenAI(api_key = openai_api_key, model="gpt-4o-mini", temperature=0)

# Pinecone
pc = Pinecone(api_key = pinecone_api_key)

# Load dataset
try:
    dataset = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")
except Exception as e:
    print(f"Error: {e}")

# Index name
index_name = "llama-2"

# Create Pinecone index if needed
try:       
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
except Exception as e:
    print(f"Error: {e}")

index = pc.Index(index_name)

# Embedding model
embed_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

# Select required columns
data_selected = dataset.select_columns(["doi", "chunk-id", "chunk", "title", "source"])
data = data_selected.to_pandas()

# Batch upload
batch_size = 100

try:
    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(i + batch_size, len(data))
        data_batched = data.iloc[i:i_end]

        ids = [f"{row['doi']}-{row['chunk-id']}" for _, row in data_batched.iterrows()]
        texts = [row["chunk"] for _, row in data_batched.iterrows()]

        embeds = embed_model.embed_documents(texts)

        metadata = [
            {
                "text": row["chunk"],
                "title": row["title"],
                "source": row["source"]
            }
            for _, row in data_batched.iterrows()
        ]

        to_upsert = list(zip(ids, embeds, metadata))
        index.upsert(vectors=to_upsert)
except Exception as e:
    print(f"Error: {e}")

# Text field
text_field = "text"

try:

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embed_model,
        text_key=text_field
    )
except Exception as e:
    print(f"Error: {e}")

# Search
query = "What is so special about Llama 2?"
results = vectorstore.similarity_search(query, top_k=3)

for r in results:
    print(r.page_content)
    print(r.metadata)

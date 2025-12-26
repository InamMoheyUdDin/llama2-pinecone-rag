
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

chat = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)
pc = Pinecone(api_key=pinecone_api_key)

dataset = load_dataset("jamescalam/llama-2-arxiv-papers-chunked", split="train")

index_name = "llama-2"

if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)

index = pc.Index(index_name)

embed_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

data = dataset.select_columns(["doi", "chunk-id", "chunk", "title", "source"]).to_pandas()

batch_size = 100
for i in tqdm(range(0, len(data), batch_size)):
    batch = data.iloc[i:i+batch_size]
    ids = [f"{r['doi']}-{r['chunk-id']}" for _, r in batch.iterrows()]
    texts = [r["chunk"] for _, r in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    metadata = [{"text": r["chunk"], "title": r["title"], "source": r["source"]} for _, r in batch.iterrows()]
    index.upsert(list(zip(ids, embeds, metadata)))

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embed_model,
    text_key="text"
)

query = "What is so special about Llama 2?"
results = vectorstore.similarity_search(query, top_k=3)

for r in results:
    print(r.page_content)
    print(r.metadata)

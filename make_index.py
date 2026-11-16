import os, time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
name = os.environ["INDEX_NAME"]

if pc.has_index(name):
    pc.delete_index(name)
    time.sleep(10)

pc.create_index(
    name=name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print(pc.describe_index(name))

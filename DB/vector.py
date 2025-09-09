from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

df = pd.read_csv("assets/data_0.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_document =  not os.path.exists(db_location)

documents = []
ids =[]

if add_document:
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"]+" "+row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

# Vector store config in chroma
vector_store = Chroma(
    collection_name="resturant-reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# adds Documents if not exist
if add_document:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k":5
    }
)
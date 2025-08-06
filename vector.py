from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="tinyllama")
db_location = "./chromadb"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["title"]+" " + row["review"],
                metadata={"title": row["title"], "author": row["author"], "rating": row["rating"]},
                id=str(i)
            )
        )
        ids.append(str(i))
        documents.append(documents)

vector_store = Chroma(
    collection_name="book_reviews",
    embedding_function=embeddings,
    persist_directory=db_location,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
    
)

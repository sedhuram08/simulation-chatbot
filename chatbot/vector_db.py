import chromadb
from .embedding import generate_embedding

# Create Chroma client
client = chromadb.PersistentClient(path="database/chromadb")

# Create or load collection
collection = client.get_or_create_collection(
    name="simulation_documents"
)


def store_chunks(chunks):
    """
    Store document chunks in ChromaDB.
    """

    for index, chunk in enumerate(chunks):

        embedding = generate_embedding(chunk)

        collection.add(
            ids=[str(index)],
            documents=[chunk],
            embeddings=[embedding]
        )


def search_chunks(query, top_k=3):

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0]
    }
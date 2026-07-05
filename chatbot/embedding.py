from sentence_transformers import SentenceTransformer

# Load the embedding model only once
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def generate_embedding(text):
    """
    Convert text into a numerical vector (embedding).
    """
    embedding = model.encode(text)

    return embedding.tolist()
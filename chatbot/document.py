import fitz
from .vector_db import store_chunks


def read_pdf(file_path):
    """
    Read the PDF and return all text.
    """
    document = fitz.open(file_path)

    text = ""

    for page in document:
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=500):
    """
    Split text into chunks.
    """
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


def process_document(file_path):
    """
    Complete indexing pipeline.
    """

    text = read_pdf(file_path)

    chunks = chunk_text(text)

    store_chunks(chunks)

    return len(chunks)
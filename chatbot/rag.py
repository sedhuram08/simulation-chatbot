from .vector_db import search_chunks
from .llm import generate_response


SIMILARITY_THRESHOLD = 0.40


SYSTEM_PROMPT = """
You are a Simulation Knowledge Assistant.

Rules:

1. Answer ONLY simulation-related questions.

2. If document context is provided,
answer ONLY from the document.

3. If no relevant document context is found,
answer using your simulation knowledge.

4. Never answer questions unrelated to simulation.

5. If the user asks unrelated questions,
reply:

'I am designed only to answer simulation-related questions.'
"""


def ask_question(question):

    search_result = search_chunks(question)

    documents = search_result["documents"]

    distances = search_result["distances"]


    if len(documents) > 0 and distances[0] < SIMILARITY_THRESHOLD:

        context = "\n\n".join(documents)

        prompt = f"""
{SYSTEM_PROMPT}

Document Context:

{context}

Question:

{question}
"""

    else:

        prompt = f"""
{SYSTEM_PROMPT}

No document contains the answer.

If the question is simulation-related,
answer using your general simulation knowledge.

Question:

{question}
"""

    answer = generate_response(prompt)

    return answer
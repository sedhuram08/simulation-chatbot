from .vector_db import search_chunks
from .llm import generate_response


SIMILARITY_THRESHOLD = 0.40


SYSTEM_PROMPT = """
You are an enterprise Simulation Knowledge Assistant.

STRICT RULES:

1. You answer ONLY simulation-related questions.

2. Before answering, determine whether the user's question is about simulation.

3. If it is NOT about simulation,
reply EXACTLY:

I am designed only to answer simulation-related questions.

Do not explain.
Do not apologize.
Do not guess.
Do not provide partial answers.

4. If document context exists,
answer ONLY from the document.

5. If no document context exists,
answer ONLY using your simulation knowledge.

6. Never answer:
- Sports
- Politics
- Movies
- Celebrities
- Programming
- Mathematics
- General Knowledge
- History
- Geography
unless the question directly relates to simulation.
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
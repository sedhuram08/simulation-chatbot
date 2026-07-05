from django.shortcuts import render
from .models import Document
from .document import process_document
from .rag import ask_question


def index(request):

    message = ""
    answer = ""

    if request.method == "POST":

        # Upload PDF
        if "pdf" in request.FILES:

            pdf = request.FILES["pdf"]

            document = Document.objects.create(
                title=pdf.name,
                file=pdf
            )

            total_chunks = process_document(document.file.path)

            message = f"Document indexed successfully ({total_chunks} chunks)."

        # Ask Question
        elif "question" in request.POST:

            question = request.POST.get("question")

            answer = ask_question(question)

    return render(request, "index.html", {
        "message": message,
        "answer": answer
    })
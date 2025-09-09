#!/usr/bin/env python3

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from DB.vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering about questions on a pizza restaurant.

Here are some relevant reviews : {reviews}

Here is a question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n------------------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n")
    if question.lower() == "q":
        break
    # Retrieve data from vector database on searching of question
    # Does SIMILARITY search
    reviews = retriever.invoke(question)

    payload = {
        "reviews": reviews,
        "question": question
    }
    results = chain.invoke(payload)
    print(results)
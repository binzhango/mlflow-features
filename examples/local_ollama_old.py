"""Minimal local example for MLflow 3 + Ollama + LangChain autolog enrichment."""

from __future__ import annotations

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def main() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("langchain-trace-enrichment")
    mlflow.langchain.autolog()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise support assistant."),
            ("human", "{question}"),
        ]
    )
    chain = (prompt | ChatOllama(model="nemotron-3-nano", temperature=0))


    result = chain.invoke({"question": "Why was invoice INV-42 charged twice?"})

    print(result.content)


if __name__ == "__main__":
    main()

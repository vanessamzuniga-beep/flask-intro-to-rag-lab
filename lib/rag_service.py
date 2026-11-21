
from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "can", "do", "does", "for", "from", "get", "how", "i",
    "if", "in", "is", "it", "me", "my", "need", "of", "on",
    "or", "our", "should", "so", "the", "their", "to", "use",
    "what", "when", "where", "who", "why", "with", "you", "your",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    tokens = {word.strip("'") for word in words}
    return {
        token for token in tokens
        if len(token) > 1 and token not in STOPWORDS
    }


def document_search_text(document: dict[str, Any]) -> str:
    tags = document.get("tags", [])
    if isinstance(tags, list):
        tags = " ".join(str(tag) for tag in tags)

    return " ".join([
        str(document.get("title", "")),
        str(document.get("category", "")),
        str(tags),
        str(document.get("text", "")),
    ])


def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    query_tokens = tokenize(query)
    document_tokens = tokenize(document_search_text(document))
    title_tokens = tokenize(str(document.get("title", "")))

    matched_terms = query_tokens.intersection(document_tokens)
    score = len(matched_terms)
    score += 0.5 * len(query_tokens.intersection(title_tokens))

    return {
        "document": document,
        "score": score,
        "matched_terms": sorted(matched_terms),
    }


def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    scored_documents = [
        score_document(query, document)
        for document in documents
    ]

    matches = [
        match for match in scored_documents
        if match["score"] >= minimum_score
    ]

    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:limit]


def format_context(context_matches: list[dict[str, Any]]) -> str:
    if not context_matches:
        return "No relevant context was found."

    formatted_documents = []

    for match in context_matches:
        document = match["document"]

        formatted_documents.append(
            f"Source ID: {document.get('id', '')}\n"
            f"Title: {document.get('title', '')}\n"
            f"Category: {document.get('category', '')}\n"
            f"Content: {document.get('text', '')}"
        )

    return "\n\n---\n\n".join(formatted_documents)


def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    context = format_context(context_matches)

    return f"""
Instructions:
You are a helpful company information assistant.
Answer the user's question using only the provided company documents.
Do not invent information or make unsupported claims.

Context:
{context}

Question:
{query}

Response requirements:
- Give a clear and helpful answer.
- Use only information supported by the context.
- If the context does not contain the answer, say that the approved
  company information is insufficient to answer.
""".strip()


def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    document = match["document"]

    return {
        "id": str(document.get("id", "")),
        "title": str(document.get("title", "")),
    }

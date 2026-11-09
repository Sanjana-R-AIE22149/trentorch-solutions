import numpy as np

def retrieval_cosine_gate(query: np.ndarray, documents: np.ndarray) -> np.ndarray:

    # Dot product of query with every document
    dot_products = documents @ query

    # Calculate norms
    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)

    # Calculate denominator
    denominator = query_norm * document_norms

    # Avoid division by zero
    similarities = np.zeros(len(documents))

    np.divide(
        dot_products,
        denominator,
        out=similarities,
        where=denominator != 0
    )

    return similarities

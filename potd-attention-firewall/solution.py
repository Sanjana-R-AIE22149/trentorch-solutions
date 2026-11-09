import numpy as np


def attention_firewall(
    q: np.ndarray, keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    q: array of shape (d,), the query for the final position.
    keys: array of shape (n, d).
    values: array of shape (n, dv).

    s_j = (q . keys[j]) / sqrt(d)
    p = softmax(s) (numerically stable, max-subtracted)
    o = sum_j p_j * values[j]

    Returns (p, o): the attention probabilities (shape (n,)) and the
    output vector (shape (dv,)).
    """
    # TODO: scale the dot products by 1/sqrt(d), softmax them stably,
    # then take the weighted sum over values.
    d = q.shape[0]

    # 1. Calculate scaled dot-product attention scores
    scores = (keys @ q) / np.sqrt(d)

    # 2. Numerically stable softmax
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    p = exp_scores / np.sum(exp_scores)

    # 3. Weighted sum of values
    o = p @ values

    return p, o

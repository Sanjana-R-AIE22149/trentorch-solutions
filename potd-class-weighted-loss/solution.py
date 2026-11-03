import numpy as np


def class_weighted_loss(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    """
    logits: n x C logits.
    labels: length-n 0-based true class indices.
    weights: length-C per-class weights.
    """
    # Stable log-softmax per row
    log_probs = logits - np.max(logits, axis=1, keepdims=True)
    log_probs -= np.log(
        np.sum(np.exp(log_probs), axis=1, keepdims=True)
    )

    # Select log-probability of each true class
    true_log_probs = log_probs[np.arange(len(labels)), labels]

    # Apply class weights and average over samples
    sample_weights = weights[labels]
    loss = -np.sum(sample_weights * true_log_probs) / len(labels)

    return float(loss)

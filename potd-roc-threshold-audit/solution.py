def roc_threshold_audit(
    probs: list[float], labels: list[int], tau: float
) -> tuple[int, int, int, int, float, float]:
    """
    probs: predicted probabilities.
    labels: binary true labels (0 or 1), same length as probs.
    tau: decision threshold. Predict positive iff p_i >= tau.

    Returns (TP, FP, TN, FN, precision, recall).
    precision = TP / (TP + FP), or 0 if that denominator is 0.
    recall = TP / (TP + FN), or 0 if that denominator is 0.
    """
    # TODO: tally the confusion matrix, then compute precision and recall.
    TP = FP = TN = FN = 0

    for p, actual in zip(probs, labels):
        predicted = 1 if p >= tau else 0

        if predicted == 1 and actual == 1:
            TP += 1
        elif predicted == 1 and actual == 0:
            FP += 1
        elif predicted == 0 and actual == 0:
            TN += 1
        else:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    return (TP, FP, TN, FN, precision, recall)

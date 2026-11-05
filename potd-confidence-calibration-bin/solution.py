def confidence_calibration_bin(
    probs: list[float], labels: list[int], k: int
) -> list[tuple[int, float, float]]:

    bins = [[] for _ in range(k)]

    # Assign each prediction to its bin
    for p, label in zip(probs, labels):
        bin_idx = min(k - 1, int(k * p))
        bins[bin_idx].append((p, label))

    result = []

    # Aggregate statistics for each bin
    for bin_data in bins:
        count = len(bin_data)

        if count == 0:
            result.append((0, 0.0, 0.0))
        else:
            mean_prob = sum(p for p, label in bin_data) / count
            positive_rate = sum(label for p, label in bin_data) / count

            result.append((count, mean_prob, positive_rate))

    return result

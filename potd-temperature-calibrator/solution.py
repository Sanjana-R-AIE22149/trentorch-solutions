import numpy as np


def temperature_calibrator(z: np.ndarray, tau: float) -> np.ndarray:
    """
    z: logits.
    tau: temperature, tau > 0.

    p = softmax(z / tau), using an overflow-safe softmax (subtract the
    max before exponentiating). Divide by tau, never multiply.

    Return p.
    """
    # TODO: scale by dividing, then stable softmax.
    scaled = z / tau

    # Stable softmax: subtract maximum before exponentiating
    shifted = scaled - np.max(scaled)
    exp_z = np.exp(shifted)

    p = exp_z / np.sum(exp_z)

    return p

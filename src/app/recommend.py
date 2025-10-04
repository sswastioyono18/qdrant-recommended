from typing import List, Dict, Any
import math

def weighted_average(vectors: List[List[float]], weights: List[float]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    wsum = 0.0
    for v, w in zip(vectors, weights):
        if len(v) != dim:
            raise ValueError("Vector dim mismatch")
        for i in range(dim):
            acc[i] += v[i] * w
        wsum += w
    if wsum == 0:
        return acc
    return [x / wsum for x in acc]

def sigmoid(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-x / k))

def final_score(pine_score: float, freshness: float = 0.5, urgency: float = 0.5, traction: float = 0.5) -> float:
    # Blend weights (tune as needed)
    return 0.6 * pine_score + 0.15 * freshness + 0.15 * urgency + 0.10 * traction

"""Binary AP@K; duplicate predictions never earn extra credit."""


def ap_at_k(actual, predicted, k=5):
    relevant = set(actual)
    if not relevant:
        return 0.0
    seen, hits, total = set(), 0, 0.0
    for rank, item in enumerate(predicted[:k], 1):
        if item in relevant and item not in seen:
            hits += 1
            total += hits / rank
        seen.add(item)
    return total / min(k, len(relevant))


def mean_ap(actual, predicted, k=5, exclude_empty=True):
    keys = [u for u in actual if actual[u] or not exclude_empty]
    if not keys:
        raise ValueError('No evaluable users')
    return sum(ap_at_k(actual[u], predicted[u], k) for u in keys) / len(keys)

def filter_support(candidates, transactions, min_sup):
    """
    Filter candidates to a frequent set by some minimum support.
    """
    counts = defaultdict(lambda: 0)
    for transaction in transactions:
        for c in (c for c in candidates if set(c).issubset(transaction)):
            counts[c] += 1
    return {i for i in candidates if counts[i]/len(transactions) >= min_sup}
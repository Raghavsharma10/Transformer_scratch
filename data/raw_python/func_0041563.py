def less_naive(gold_schemes):
    """find 'less naive' baseline (most common scheme of a given length in subcorpus)"""
    best_schemes = defaultdict(lambda: defaultdict(int))
    for g in gold_schemes:
        best_schemes[len(g)][tuple(g)] += 1

    for i in best_schemes:
        best_schemes[i] = tuple(max(best_schemes[i].items(), key=lambda x: x[1])[0])

    naive_schemes = []
    for g in gold_schemes:
        naive_schemes.append(best_schemes[len(g)])
    return naive_schemes
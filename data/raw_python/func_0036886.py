def generate_candidates(freq_set, k):
    """
    Generate candidates for an iteration.
    Use this only for k >= 2.
    """
    single_set = {(i,) for i in set(flatten(freq_set))}

    # TO DO generating all combinations gets very slow for large documents.
    # Is there a way of doing this without exhaustively searching all combinations?
    cands = [flatten(f) for f in combinations(single_set, k)]
    return [cand for cand in cands if validate_candidate(cand, freq_set, k)]
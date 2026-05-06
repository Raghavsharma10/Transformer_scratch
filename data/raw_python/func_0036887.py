def validate_candidate(candidate, freq_set, k):
    """
    Checks if we should keep a candidate.
    We keep a candidate if all its k-1-sized subsets
    are present in the frequent sets.
    """
    for subcand in combinations(candidate, k-1):
        if subcand not in freq_set:
            return False
    return True
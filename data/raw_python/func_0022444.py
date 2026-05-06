def merge_sims(oldsims, newsims, clip=None):
    """Merge two precomputed similarity lists, truncating the result to `clip` most similar items."""
    if oldsims is None:
        result = newsims or []
    elif newsims is None:
        result = oldsims
    else:
        result = sorted(oldsims + newsims, key=lambda item: -item[1])
    if clip is not None:
        result = result[:clip]
    return result
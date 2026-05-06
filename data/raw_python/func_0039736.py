def sort(base, items):
    """Get a sorted list of items ranked in descending similarity.

    @param base: base item to perform comparison against
    @param items: list of items to compare to the base
    @return: list of items sorted by similarity to the base

    """
    return sorted(items, key=base.similarity, reverse=True)
def duplicates(base, items):
    """Get an iterator of items similar but not equal to the base.

    @param base: base item to perform comparison against
    @param items: list of items to compare to the base
    @return: generator of items sorted by similarity to the base

    """
    for item in items:
        if item.similarity(base) and not item.equality(base):
            yield item
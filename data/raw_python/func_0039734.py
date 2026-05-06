def match_similar(base, items):
    """Get the most similar matching item from a list of items.

    @param base: base item to locate best match
    @param items: list of items for comparison
    @return: most similar matching item or None

    """
    finds = list(find_similar(base, items))
    if finds:
        return max(finds, key=base.similarity)  # TODO: make O(n)

    return None
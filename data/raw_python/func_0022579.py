def dedupe(items):
    """Remove duplicates from a sequence (of hashable items) while maintaining
    order. NOTE: This only works if items in the list are hashable types.

    Taken from the Python Cookbook, 3rd ed. Such a great book!

    """
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)
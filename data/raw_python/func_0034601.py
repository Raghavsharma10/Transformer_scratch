def walk_tree(start, attr):
    """
    Recursively walk through a tree relationship. This iterates a tree in a top-down approach,
    fully reaching the end of a lineage before moving onto the next sibling of that generation.
    """
    path = [start]
    for child in path:
        yield child
        idx = path.index(child)
        for grandchild in reversed(getattr(child, attr)):
            path.insert(idx + 1, grandchild)
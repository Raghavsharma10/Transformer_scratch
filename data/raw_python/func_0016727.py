def walk_subclasses(root):
    """Does not yield the input class"""
    classes = [root]
    visited = set()
    while classes:
        cls = classes.pop()
        if cls is type or cls in visited:
            continue
        classes.extend(cls.__subclasses__())
        visited.add(cls)
        if cls is not root:
            yield cls
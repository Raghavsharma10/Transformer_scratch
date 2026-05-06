def get_subclass_tree(cls, ensure_unique=True):
    """Returns all subclasses (direct and recursive) of cls."""
    subclasses = []
    # cls.__subclasses__() fails on classes inheriting from type
    for subcls in type.__subclasses__(cls):
        subclasses.append(subcls)
        subclasses.extend(get_subclass_tree(subcls, ensure_unique))
    return list(set(subclasses)) if ensure_unique else subclasses
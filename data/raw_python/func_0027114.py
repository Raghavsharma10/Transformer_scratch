def dict_to_object(source):
    """Returns an object with the key-value pairs in source as attributes."""
    target = inspectable_class.InspectableClass()
    for k, v in source.items():
        setattr(target, k, v)
    return target
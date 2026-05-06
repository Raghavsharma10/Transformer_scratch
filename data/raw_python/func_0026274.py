def find_class_in_list(klass, lst):
    """
    Returns the first occurrence of an instance of type `klass` in 
    the given list, or None if no such instance is present.
    """
    filtered = list(filter(lambda x: x.__class__ == klass, lst))
    if filtered:
        return filtered[0]
    return None
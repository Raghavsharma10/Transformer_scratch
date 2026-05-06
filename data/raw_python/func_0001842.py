def dict_of_sets_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a set in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to set in dictionary

    Returns:
        None

    """
    set_objs = dictionary.get(key, set())
    set_objs.add(value)
    dictionary[key] = set_objs
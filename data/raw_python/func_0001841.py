def dict_of_lists_add(dictionary, key, value):
    # type: (DictUpperBound, Any, Any) -> None
    """Add value to a list in a dictionary by key

    Args:
        dictionary (DictUpperBound): Dictionary to which to add values
        key (Any): Key within dictionary
        value (Any): Value to add to list in dictionary

    Returns:
        None

    """
    list_objs = dictionary.get(key, list())
    list_objs.append(value)
    dictionary[key] = list_objs
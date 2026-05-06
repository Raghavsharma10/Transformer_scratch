def flatten(lol):
    """Flatten a list of lists to a list.

    :param lol: A list of lists in arbitrary depth.
    :type lol: list of list.

    :returns: flat list of elements.
    """
    new_list = []
    for element in lol:
        if element is None:
            continue
        elif not isinstance(element, list) and not isinstance(element, tuple):
            new_list.append(element)
        elif len(element) > 0:
            new_list.extend(flatten(element))
    return new_list
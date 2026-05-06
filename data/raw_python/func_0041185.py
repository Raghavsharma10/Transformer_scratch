def force_list(element):
    """
    Given an element or a list, concatenates every element and clean it to
    create a full text
    """
    if element is None:
        return []

    if isinstance(element, (collections.Iterator, list)):
        return element

    return [element]
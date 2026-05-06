def remove_all_attributes(element, exclude=None):
    """
    This method will remove all attributes of any provided element.

    A list of strings may be passed to the keyward-argument "exclude", which
    will serve as a list of attributes which will not be removed.
    """
    if exclude is None:
        exclude = []
    for k in element.attrib.keys():
        if k not in exclude:
            element.attrib.pop(k)
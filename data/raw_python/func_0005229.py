def _xml_element_value(el: Element, int_tags: list):
    """
    Gets XML Element value.
    :param el: Element
    :param int_tags: List of tags that should be treated as ints
    :return: value of the element (int/str)
    """
    # None
    if el.text is None:
        return None
    # int
    try:
        if el.tag in int_tags:
            return int(el.text)
    except:
        pass
    # default to str if not empty
    s = str(el.text).strip()
    return s if s else None
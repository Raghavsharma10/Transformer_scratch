def _most_restrictive(date_elems):
    """
    Return the date_elem that has the most restrictive range from date_elems
    """
    most_index = len(DATE_ELEMENTS)
    for date_elem in date_elems:
        if date_elem in DATE_ELEMENTS and DATE_ELEMENTS.index(date_elem) < most_index:
            most_index = DATE_ELEMENTS.index(date_elem)
    if most_index < len(DATE_ELEMENTS):
        return DATE_ELEMENTS[most_index]
    else:
        raise KeyError('No least restrictive date element found')
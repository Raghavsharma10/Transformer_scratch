def split_input_references(to_split):
    """
    Returns the given string in normal strings and unresolved input references.
    An input reference is identified as something of the following form $(...).

    Example:
    split_input_reference("a$(b)cde()$(fg)") == ["a", "$(b)", "cde()", "$(fg)"]

    :param to_split: The string to split
    :raise InvalidInputReference: If an input reference is not closed and a new reference starts or the string ends.
    :return: A list of normal strings and unresolved input references.
    """
    parts = partition_all(to_split, [INPUT_REFERENCE_START, INPUT_REFERENCE_END])

    result = []
    part = []
    in_reference = False
    for p in parts:
        if in_reference:
            if p == INPUT_REFERENCE_START:
                raise InvalidInputReference('A new input reference has been started, although the old input reference'
                                            'has not yet been completed.\n{}'.format(to_split))
            elif p == ")":
                part.append(")")
                result.append(''.join(part))
                part = []
                in_reference = False
            else:
                part.append(p)
        else:
            if p == INPUT_REFERENCE_START:
                if part:
                    result.append(''.join(part))
                part = [INPUT_REFERENCE_START]
                in_reference = True
            else:
                part.append(p)

    if in_reference:
        raise InvalidInputReference('Input reference not closed.\n{}'.format(to_split))
    elif part:
        result.append(''.join(part))

    return result
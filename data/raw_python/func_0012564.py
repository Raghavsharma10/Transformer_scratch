def int_to_roman(integer):
    """
    Convert an integer into a string of roman numbers.

    .. code: python

        reusables.int_to_roman(445)
        # 'CDXLV'


    :param integer:
    :return: roman string
    """
    if not isinstance(integer, int):
        raise ValueError("Input integer must be of type int")
    output = []
    while integer > 0:
        for r, i in sorted(_roman_dict.items(),
                           key=lambda x: x[1], reverse=True):
            while integer >= i:
                output.append(r)
                integer -= i
    return "".join(output)
def num_equal(result, operator, comparator):
    """
    Returns the number of elements in a list that pass a comparison

    :param result: The list of results of a dice roll
    :param operator: Operator in string to perform comparison on:
        Either '+', '-', or '*'
    :param comparator: The value to compare
    :return:
    """
    if operator == '<':
        return len([x for x in result if x < comparator])
    elif operator == '>':
        return len([x for x in result if x > comparator])
    elif operator == '=':
        return len([x for x in result if x == comparator])
    else:
        raise ValueError
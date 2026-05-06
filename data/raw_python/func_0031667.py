def expand_recurring(number, repeat=5):
    """
    Expands a recurring pattern within a number.

    Args:
        number(tuple): the number to process in the form:
            (int, int, int, ... ".", ... , int int int)
        repeat: the number of times to expand the pattern.

    Returns:
        The original number with recurring pattern expanded.

    Example:
        >>> expand_recurring((1, ".", 0, "[", 9, "]"), repeat=3)
        (1, '.', 0, 9, 9, 9, 9)
    """
    if "[" in number:
        pattern_index = number.index("[")
        pattern = number[pattern_index + 1:-1]
        number = number[:pattern_index]
        number = number + pattern * (repeat + 1)
    return number
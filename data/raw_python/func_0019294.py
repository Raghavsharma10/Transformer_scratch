def print_values(values, width=70):
    """Print the given values in multiple lines with a certain maximum width.

    By default, each line contains at most 70 characters:

    >>> from hydpy import print_values
    >>> print_values(range(21))
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20

    You can change this default behaviour by passing an alternative
    number of characters:

    >>> print_values(range(21), width=30)
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20
    """
    for line in textwrap.wrap(repr_values(values), width=width):
        print(line)
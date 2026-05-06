def binboolflip(item):
    """
    Convert 0 or 1 to False or True (or vice versa).
    The converter works as follows:

    - 0 > False
    - False > 0
    - 1 > True
    - True > 1

    :type item: integer or boolean
    :param item: The item to convert.

    >>> binboolflip(0)
    False

    >>> binboolflip(False)
    0

    >>> binboolflip(1)
    True

    >>> binboolflip(True)
    1

    >>> binboolflip("foo")
    Traceback (most recent call last):
      ...
    ValueError: Invalid item specified.
    """

    if item in [0, False, 1, True]:
        return int(item) if isinstance(item, bool) else bool(item)

    # Raise a warning
    raise ValueError("Invalid item specified.")
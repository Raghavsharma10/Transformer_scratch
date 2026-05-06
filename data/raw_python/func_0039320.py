def pykeyword(operation='list', keywordtotest=None):
    """
    Check if a keyword exists in the Python keyword dictionary.

    :type operation: string
    :param operation: Whether to list or check the keywords. Possible options are 'list' and 'in'.

    :type keywordtotest: string
    :param keywordtotest: The keyword to check.

    :return: The list of keywords or if a keyword exists.
    :rtype: list or boolean

    >>> "True" in pykeyword("list")
    True

    >>> pykeyword("in", "True")
    True

    >>> pykeyword("in", "foo")
    False

    >>> pykeyword("foo", "foo")
    Traceback (most recent call last):
      ...
    ValueError: Invalid operation specified.
    """

    # If the operation was 'list'
    if operation == 'list':
        # Return an array of keywords
        return str(keyword.kwlist)

    # If the operation was 'in'
    elif operation == 'in':
        # Return a boolean for if the string was a keyword
        return keyword.iskeyword(str(keywordtotest))

    # Raise a warning
    raise ValueError("Invalid operation specified.")
def splitstring(string, splitcharacter=' ', part=None):
    """
    Split a string based on a character and get the parts as a list.

    :type string: string
    :param string: The string to split.

    :type splitcharacter: string
    :param splitcharacter: The character to split for the string.

    :type part: integer
    :param part: Get a specific part of the list.

    :return: The split string or a specific part of it
    :rtype: list or string

    >>> splitstring('hello world !')
    ['hello', 'world', '!']

    >>> splitstring('hello world !', ' ', None)
    ['hello', 'world', '!']

    >>> splitstring('hello world !', ' ', None)
    ['hello', 'world', '!']

    >>> splitstring('hello world !', ' ', 0)
    'hello'

    """

    # If the part is empty
    if part in [None, '']:
        # Return an array of the splitted text
        return str(string).split(splitcharacter)

    # Return an array of the splitted text with a specific part
    return str(string).split(splitcharacter)[part]
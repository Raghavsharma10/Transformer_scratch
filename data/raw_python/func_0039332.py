def spacelist(listtospace, spacechar=" "):
    """
    Convert a list to a string with all of the list's items spaced out.

    :type listtospace: list
    :param listtospace: The list to space out.

    :type spacechar: string
    :param spacechar: The characters to insert between each list item. Default is: " ".
    """
    output = ''
    space = ''
    output += str(listtospace[0])
    space += spacechar
    for listnum in range(1, len(listtospace)):
        output += space
        output += str(listtospace[listnum])
    return output
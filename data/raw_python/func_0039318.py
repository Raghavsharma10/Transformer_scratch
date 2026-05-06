def shellinput(initialtext='>> ', splitpart=' '):
    """
    Give the user a shell-like interface to enter commands which
    are returned as a multi-part list containing the command
    and each of the arguments.

    :type initialtext: string
    :param initialtext: Set the text to be displayed as the prompt.

    :type splitpart: string
    :param splitpart: The character to split when generating the list item.

    :return: A string of the user's input or a list of the user's input split by the split character.
    :rtype: string or list
    """

    # Get the user's input
    shelluserinput = input(str(initialtext))

    # Return the computed result
    return shelluserinput if splitpart in (
        '', None) else shelluserinput.split(splitpart)
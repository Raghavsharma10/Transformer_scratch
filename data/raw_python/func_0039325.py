def convertbinary(value, argument):
    """
    Convert text to binary form or backwards.

    :type value: string
    :param value: The text or the binary text

    :type argument: string
    :param argument: The action to perform on the value. Can be "to" or "from".
    """

    if argument == 'to':
        return bin(value)
    elif argument == 'from':
        return format(value)
    raise ValueError("Invalid argument specified.")
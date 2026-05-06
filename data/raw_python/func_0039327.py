def convertascii(value, command='to'):
    """
    Convert an ASCII value to a symbol

    :type value: string
    :param value: The text or the text in ascii form.

    :type argument: string
    :param argument: The action to perform on the value. Can be "to" or "from".
    """
    command = command.lower()
    if command == 'to':
        return chr(value)
    elif command == 'from':
        return ord(value)
    else:
        raise ValueError('Invalid operation provided.')
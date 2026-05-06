def availchars(charactertype):
    """
    Get all the available characters for a specific type.

    :type charactertype: string
    :param charactertype: The characters to get. Can be 'letters', 'lowercase, 'uppercase', 'digits', 'hexdigits', 'punctuation', 'printable', 'whitespace' or 'all'.

    >>> availchars("lowercase")
    'abcdefghijklmnopqrstuvwxyz'
    """

    # If the lowercase version of the character type is 'letters'
    if charactertype.lower() == 'letters':
        # Return the result
        return string.ascii_letters

    # If the lowercase version of the character type is 'lowercase'
    elif charactertype.lower() == 'lowercase':
        # Return the result
        return string.ascii_lowercase

    # If the lowercase version of the character type is 'uppercase'
    elif charactertype.lower() == 'uppercase':
        # Return the result
        return string.ascii_uppercase

    # If the lowercase version of the character type is 'digits'
    elif charactertype.lower() == 'digits':
        # Return the result
        return string.digits

    # If the lowercase version of the character type is 'hexdigits'
    elif charactertype.lower() == 'hexdigits':
        # Return the result
        return string.hexdigits

    # If the lowercase version of the character type is 'punctuation'
    elif charactertype.lower() == 'punctuation':
        # Return the result
        return string.punctuation

    # If the lowercase version of the character type is 'printable'
    elif charactertype.lower() == 'printable':
        # Return the result
        return string.printable

    # If the lowercase version of the character type is 'whitespace'
    elif charactertype.lower() == 'whitespace':
        # Return the result
        return string.whitespace

    # If the lowercase version of the character type is 'all'
    elif charactertype.lower() == 'all':
        # Return the result
        return string.ascii_letters + string.ascii_lowercase + string.ascii_uppercase + string.digits + string.hexdigits + string.punctuation + string.printable + string.whitespace

    # Raise a warning
    raise ValueError("Invalid character type provided.")
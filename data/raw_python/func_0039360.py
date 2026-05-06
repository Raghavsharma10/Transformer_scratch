def case(text, casingformat='sentence'):
    """
    Change the casing of some text.

    :type text: string
    :param text: The text to change the casing of.

    :type casingformat: string
    :param casingformat: The format of casing to apply to the text. Can be 'uppercase', 'lowercase', 'sentence' or 'caterpillar'.

    :raises ValueError: Invalid text format specified.

    >>> case("HELLO world", "uppercase")
    'HELLO WORLD'
    """

    # If the lowercase version of the casing format is 'uppercase'
    if casingformat.lower() == 'uppercase':
        # Return the uppercase version
        return str(text.upper())

    # If the lowercase version of the casing format is 'lowercase'
    elif casingformat.lower() == 'lowercase':
        # Return the lowercase version
        return str(text.lower())

    # If the lowercase version of the casing format is 'sentence'
    elif casingformat.lower() == 'sentence':
        # Return the sentence case version
        return str(text[0].upper()) + str(text[1:])

    # If the lowercase version of the casing format is 'caterpillar'
    elif casingformat.lower() == 'caterpillar':
        # Return the caterpillar case version
        return str(text.lower().replace(" ", "_"))

    # Raise a warning
    raise ValueError("Invalid text format specified.")
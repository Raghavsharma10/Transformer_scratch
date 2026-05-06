def get_accent_char(char):
    """
    Get the accent of an single char, if any.
    """
    index = utils.VOWELS.find(char.lower())
    if (index != -1):
        return 5 - index % 6
    else:
        return Accent.NONE
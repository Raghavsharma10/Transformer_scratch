def remove_accent_string(string):
    """
    Remove all accent from a whole string.
    """
    return utils.join([add_accent_char(c, Accent.NONE) for c in string])
def add_accent_at(string, accent, index):
    """
    Add mark to the index-th character of the given string.  Return
    the new string after applying change.
    (unused)
    """
    if index == -1:
        return string
    # Python can handle the case which index is out of range of given string
    return string[:index] + \
        accent.accent.add_accent_char(string[index], accent) + \
        string[index+1:]
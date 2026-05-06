def add_accent_char(char, accent):
    """
    Add accent to a single char.  Parameter accent is member of class
    Accent
    """
    if char == "":
        return ""
    case = char.isupper()
    char = char.lower()
    index = utils.VOWELS.find(char)
    if (index != -1):
        index = index - index % 6 + 5
        char = utils.VOWELS[index - accent]
    return utils.change_case(char, case)
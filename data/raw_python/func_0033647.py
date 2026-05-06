def add_mark_char(char, mark):
    """
    Add mark to a single char.
    """
    if char == "":
        return ""
    case = char.isupper()
    ac = accent.get_accent_char(char)
    char = accent.add_accent_char(char.lower(), Accent.NONE)
    new_char = char
    if mark == Mark.HAT:
        if char in FAMILY_A:
            new_char = "â"
        elif char in FAMILY_O:
            new_char = "ô"
        elif char in FAMILY_E:
            new_char = "ê"
    elif mark == Mark.HORN:
        if char in FAMILY_O:
            new_char = "ơ"
        elif char in FAMILY_U:
            new_char = "ư"
    elif mark == Mark.BREVE:
        if char in FAMILY_A:
            new_char = "ă"
    elif mark == Mark.BAR:
        if char in FAMILY_D:
            new_char = "đ"
    elif mark == Mark.NONE:
        if char in FAMILY_A:
            new_char = "a"
        elif char in FAMILY_E:
            new_char = "e"
        elif char in FAMILY_O:
            new_char = "o"
        elif char in FAMILY_U:
            new_char = "u"
        elif char in FAMILY_D:
            new_char = "d"

    new_char = accent.add_accent_char(new_char, ac)
    return utils.change_case(new_char, case)
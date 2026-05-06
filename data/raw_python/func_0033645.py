def get_mark_char(char):
    """
    Get the mark of a single char, if any.
    """
    char = accent.remove_accent_char(char.lower())
    if char == "":
        return Mark.NONE
    if char == "đ":
        return Mark.BAR
    if char in "ă":
        return Mark.BREVE
    if char in "ơư":
        return Mark.HORN
    if char in "âêô":
        return Mark.HAT
    return Mark.NONE
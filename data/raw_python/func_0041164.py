def select_regexp_char(char):
    """
    Select correct regex depending the char
    """
    regexp = '{}'.format(char)

    if not isinstance(char, str) and not isinstance(char, int):
        regexp = ''

    if isinstance(char, str) and not char.isalpha() and not char.isdigit():
        regexp = r"\{}".format(char)

    return regexp
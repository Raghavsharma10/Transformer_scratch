def tuple_to_qfont(tup):
    """
    Create a QFont from tuple:
        (family [string], size [int], italic [bool], bold [bool])
    """
    if not isinstance(tup, tuple) or len(tup) != 4 \
       or not is_text_string(tup[0]) \
       or not isinstance(tup[1], int) \
       or not isinstance(tup[2], bool) \
       or not isinstance(tup[3], bool):
        return None
    font = QFont()
    family, size, italic, bold = tup
    font.setFamily(family)
    font.setPointSize(size)
    font.setItalic(italic)
    font.setBold(bold)
    return font
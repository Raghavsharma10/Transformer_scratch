def is_jamo(character):
    """Test if a single character is a jamo character.
    Valid jamo includes all modern and archaic jamo, as well as all HCJ.
    Non-assigned code points are invalid.
    """
    code = ord(character)
    return 0x1100 <= code <= 0x11FF or\
        0xA960 <= code <= 0xA97C or\
        0xD7B0 <= code <= 0xD7C6 or 0xD7CB <= code <= 0xD7FB or\
        is_hcj(character)
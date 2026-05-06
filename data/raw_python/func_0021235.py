def is_jamo_modern(character):
    """Test if a single character is a modern jamo character.
    Modern jamo includes all U+11xx jamo in addition to HCJ in modern usage,
    as defined in Unicode 7.0.
    WARNING: U+1160 is NOT considered a modern jamo character, but it is listed
    under 'Medial Vowels' in the Unicode 7.0 spec.
    """
    code = ord(character)
    return 0x1100 <= code <= 0x1112 or\
        0x1161 <= code <= 0x1175 or\
        0x11A8 <= code <= 0x11C2 or\
        is_hcj_modern(character)
def is_jamo_compound(character):
    """Test if a single character is a compound, i.e., a consonant
    cluster, double consonant, or dipthong.
    """
    if len(character) != 1:
        return False
        # Consider instead:
        # raise TypeError('is_jamo_compound() expected a single character')
    if is_jamo(character):
        return character in JAMO_COMPOUNDS
    return False
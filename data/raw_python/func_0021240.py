def jamo_to_hangul(lead, vowel, tail=''):
    """Return the Hangul character for the given jamo input.
    Integers corresponding to U+11xx jamo codepoints, U+11xx jamo characters,
    or HCJ are valid inputs.

    Outputs a one-character Hangul string.

    This function is identical to j2h.
    """
    # Internally, we convert everything to a jamo char,
    # then pass it to _jamo_to_hangul_char
    lead = hcj_to_jamo(lead, "lead")
    vowel = hcj_to_jamo(vowel, "vowel")
    if not tail or ord(tail) == 0:
        tail = None
    elif is_hcj(tail):
        tail = hcj_to_jamo(tail, "tail")
    if (is_jamo(lead) and get_jamo_class(lead) == "lead") and\
       (is_jamo(vowel) and get_jamo_class(vowel) == "vowel") and\
       ((not tail) or (is_jamo(tail) and get_jamo_class(tail) == "tail")):
        result = _jamo_to_hangul_char(lead, vowel, tail)
        if is_hangul_char(result):
            return result
    raise InvalidJamoError("Could not synthesize characters to Hangul.",
                           '\x00')
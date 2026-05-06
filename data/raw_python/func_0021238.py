def hcj_to_jamo(hcj_char, position="vowel"):
    """Convert a HCJ character to a jamo character.
    Arguments may be single characters along with the desired jamo class
    (lead, vowel, tail). Non-mappable input will raise an InvalidJamoError.
    """
    if position == "lead":
        jamo_class = "CHOSEONG"
    elif position == "vowel":
        jamo_class = "JUNGSEONG"
    elif position == "tail":
        jamo_class = "JONGSEONG"
    else:
        raise InvalidJamoError("No mapping from input to jamo.", hcj_char)
    jamo_name = re.sub("(?<=HANGUL )(\w+)",
                       jamo_class,
                       _get_unicode_name(hcj_char))
    # TODO: add tests that test non entries.
    if jamo_name in _JAMO_REVERSE_LOOKUP.keys():
        return _JAMO_REVERSE_LOOKUP[jamo_name]
    return hcj_char
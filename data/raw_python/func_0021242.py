def compose_jamo(*parts):
    """Return the compound jamo for the given jamo input.
    Integers corresponding to U+11xx jamo codepoints, U+11xx jamo
    characters, or HCJ are valid inputs.

    Outputs a one-character jamo string.
    """
    # Internally, we convert everything to a jamo char,
    # then pass it to _jamo_to_hangul_char
    # NOTE: Relies on hcj_to_jamo not strictly requiring "position" arg.
    for p in parts:
        if not (type(p) == str and len(p) == 1 and 2 <= len(parts) <= 3):
            raise TypeError("compose_jamo() expected 2-3 single characters " +
                            "but received " + str(parts),
                            '\x00')
    hcparts = [j2hcj(_) for _ in parts]
    hcparts = tuple(hcparts)
    if hcparts in _COMPONENTS_REVERSE_LOOKUP:
        return _COMPONENTS_REVERSE_LOOKUP[hcparts]
    raise InvalidJamoError(
            "Could not synthesize characters to compound: " + ", ".join(
                    str(_) + "(U+" + str(hex(ord(_)))[2:] +
                    ")" for _ in parts), '\x00')
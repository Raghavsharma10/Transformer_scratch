def _get_unicode_name(char):
    """Fetch the unicode name for jamo characters.
    """
    if char not in _JAMO_TO_NAME.keys() and char not in _HCJ_TO_NAME.keys():
        raise InvalidJamoError("Not jamo or nameless jamo character", char)
    else:
        if is_hcj(char):
            return _HCJ_TO_NAME[char]
        return _JAMO_TO_NAME[char]
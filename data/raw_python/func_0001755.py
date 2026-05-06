def iso_639_alpha3(code):
    """Convert a given language identifier into an ISO 639 Part 2 code, such
    as "eng" or "deu". This will accept language codes in the two- or three-
    letter format, and some language names. If the given string cannot be
    converted, ``None`` will be returned.
    """
    code = normalize_code(code)
    code = ISO3_MAP.get(code, code)
    if code in ISO3_ALL:
        return code
def list_to_alpha3(languages, synonyms=True):
    """Parse all the language codes in a given list into ISO 639 Part 2 codes
    and optionally expand them with synonyms (i.e. other names for the same
    language)."""
    codes = set([])
    for language in ensure_list(languages):
        code = iso_639_alpha3(language)
        if code is None:
            continue
        codes.add(code)
        if synonyms:
            codes.update(expand_synonyms(code))
    return codes
def identify(text):
    """Identify whether a string is simplified or traditional Chinese.

    Returns:
        None: if there are no recognizd Chinese characters.
        EITHER: if the test is inconclusive.
        TRAD: if the text is traditional.
        SIMP: if the text is simplified.
        BOTH: the text has characters recognized as being solely traditional
            and other characters recognized as being solely simplified.

    """
    filtered_text = set(list(text)).intersection(ALL_CHARS)
    if len(filtered_text) is 0:
        return None
    if filtered_text.issubset(SHARED_CHARS):
        return EITHER
    if filtered_text.issubset(TRAD_CHARS):
        return TRAD
    if filtered_text.issubset(SIMP_CHARS):
        return SIMP
    if filtered_text.difference(TRAD_CHARS).issubset(SIMP_CHARS):
        return BOTH
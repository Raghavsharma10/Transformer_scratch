def decompose_nfkd(text):
    """Perform unicode compatibility decomposition.

    This will replace some non-standard value representations in unicode and
    normalise them, while also separating characters and their diacritics into
    two separate codepoints.
    """
    if text is None:
        return None
    if not hasattr(decompose_nfkd, '_tr'):
        decompose_nfkd._tr = Transliterator.createInstance('Any-NFKD')
    return decompose_nfkd._tr.transliterate(text)
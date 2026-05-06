def invalid_ipa_characters(unicode_string, indices=False):
    """
    Return the list of Unicode characters
    in the given Unicode string
    that are not IPA valid.

    Return ``None`` if ``unicode_string`` is ``None``.

    :param str unicode_string: the Unicode string to be parsed
    :param bool indices: if ``True``, return a list of pairs (index, invalid character),
                         instead of a list of str (characters).
    :rtype: list of str or list of (int, str) 
    """
    if unicode_string is None:
        return None
    if indices:
        return [(i, unicode_string[i]) for i in range(len(unicode_string)) if unicode_string[i] not in UNICODE_TO_IPA]
    return set([c for c in unicode_string if c not in UNICODE_TO_IPA])
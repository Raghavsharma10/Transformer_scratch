def ipa_substrings(unicode_string, single_char_parsing=False):
    """
    Return a list of (non-empty) substrings of the given string,
    where each substring is either:
    
    1. the longest Unicode string starting at the current index
       representing a (known) valid IPA character, or
    2. a single Unicode character (which is not IPA valid).

    If ``single_char_parsing`` is ``False``,
    parse the string one Unicode character at a time,
    that is, do not perform the greedy parsing.

    For example, if ``s = u"\u006e\u0361\u006d"``,
    with ``single_char_parsing=True`` the result will be
    a list with a single element: ``[u"\u006e\u0361\u006d"]``,
    while ``single_char_parsing=False`` will yield a list with three elements:
    ``[u"\u006e", u"\u0361", u"\u006d"]``.

    Return ``None`` if ``unicode_string`` is ``None``.

    :param str unicode_string: the Unicode string to be parsed
    :param bool single_char_parsing: if ``True``, parse one Unicode character at a time
    :rtype: list of str
    """
    return split_using_dictionary(
        string=unicode_string,
        dictionary=UNICODE_TO_IPA,
        max_key_length=UNICODE_TO_IPA_MAX_KEY_LENGTH,
        single_char_parsing=single_char_parsing
    )
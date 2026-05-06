def remove_invalid_ipa_characters(unicode_string, return_invalid=False, single_char_parsing=False):
    """
    Remove all Unicode characters that are not IPA valid
    from the given string,
    and return a list of substrings of the given string,
    each mapping to a (known) valid IPA character.

    Return ``None`` if ``unicode_string`` is ``None``.

    :param str unicode_string: the Unicode string to be parsed
    :param bool return_invalid: if ``True``, return a pair ``(valid, invalid)``,
                                where ``invalid`` is a list of Unicode characters
                                that are not IPA valid.
    :param bool single_char_parsing: if ``True``, parse one Unicode character at a time
    :rtype: list of str
    """
    if unicode_string is None:
        return None
    substrings = ipa_substrings(unicode_string, single_char_parsing=single_char_parsing)
    valid = [s for s in substrings if s in UNICODE_TO_IPA]
    if return_invalid:
        return (valid, [s for s in substrings if s not in UNICODE_TO_IPA])
    return valid
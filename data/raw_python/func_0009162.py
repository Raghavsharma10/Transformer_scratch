def to_ipa(s, delimiter=' ', all_readings=False, container='[]'):
    """Convert a string's Chinese characters to IPA.

    *s* is a string containing Chinese characters.

    *delimiter* is the character used to indicate word boundaries in *s*.
    This is used to differentiate between words and characters so that a more
    accurate reading can be returned.

    *all_readings* is a boolean value indicating whether or not to return all
    possible readings in the case of words/characters that have multiple
    readings. *container* is a two character string that is used to
    enclose words/characters if *all_readings* is ``True``. The default
    ``'[]'`` is used like this: ``'[READING1/READING2]'``.

    Characters not recognized as Chinese are left untouched.

    """
    numbered_pinyin = to_pinyin(s, delimiter, all_readings, container, False)
    ipa = pinyin_to_ipa(numbered_pinyin)
    return ipa
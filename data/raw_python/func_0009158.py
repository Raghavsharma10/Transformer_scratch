def _hanzi_to_pinyin(hanzi):
    """Return the Pinyin reading for a Chinese word.

    If the given string *hanzi* matches a CC-CEDICT word, the return value is
    formatted like this: [WORD_READING1, WORD_READING2, ...]

    If the given string *hanzi* doesn't match a CC-CEDICT word, the return
    value is formatted like this: [[CHAR_READING1, CHAR_READING2 ...], ...]

    When returning character readings, if a character wasn't recognized, the
    original character is returned, e.g. [[CHAR_READING1, ...], CHAR, ...]

    """
    try:
        return _HANZI_PINYIN_MAP['words'][hanzi]
    except KeyError:
        return [_CHARACTERS.get(character, character) for character in hanzi]
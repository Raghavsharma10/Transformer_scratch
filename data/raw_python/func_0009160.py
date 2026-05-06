def to_pinyin(s, delimiter=' ', all_readings=False, container='[]',
              accented=True):
    """Convert a string's Chinese characters to Pinyin readings.

    *s* is a string containing Chinese characters. *accented* is a
    boolean value indicating whether to return accented or numbered Pinyin
    readings.

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
    hanzi = s
    pinyin = ''

    # Process the given string.
    while hanzi:

        # Get the next match in the given string.
        match = re.search('[^%s%s]+' % (delimiter, zhon.hanzi.punctuation),
                          hanzi)

        # There are no more matches, but the string isn't finished yet.
        if match is None and hanzi:
            pinyin += hanzi
            break

        match_start, match_end = match.span()

        # Process the punctuation marks that occur before the match.
        if match_start > 0:
            pinyin += hanzi[0:match_start]

        # Get the Chinese word/character readings.
        readings = _hanzi_to_pinyin(match.group())

        # Process the returned word readings.
        if match.group() in _WORDS:
            if all_readings:
                reading = _enclose_readings(container,
                                            _READING_SEPARATOR.join(readings))
            else:
                reading = readings[0]
            pinyin += reading

        # Process the returned character readings.
        else:
            # Process each character individually.
            for character in readings:
                # Don't touch unrecognized characters.
                if isinstance(character, str):
                    pinyin += character
                # Format multiple readings.
                elif isinstance(character, list) and all_readings:
                    pinyin += _enclose_readings(
                        container, _READING_SEPARATOR.join(character))
                # Select and format the most common reading.
                elif isinstance(character, list) and not all_readings:
                    # Add an apostrophe to separate syllables.
                    if (pinyin and character[0][0] in zhon.pinyin.vowels and
                            pinyin[-1] in zhon.pinyin.lowercase):
                        pinyin += "'"
                    pinyin += character[0]

        # Move ahead in the given string.
        hanzi = hanzi[match_end:]

    if accented:
        return pinyin
    else:
        return accented_to_numbered(pinyin)
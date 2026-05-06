def _pinyin_generator(chars, format):
    """Generate pinyin for chars, if char is not chinese character,
    itself will be returned.
    Chars must be unicode list.
    """
    for char in chars:
        key = "%X" % ord(char)
        pinyin = pinyin_dict.get(key, char)
        tone = pinyin_tone.get(key, 0)

        if tone == 0 or format == "strip":
            pass
        elif format == "numerical":
            pinyin += str(tone)
        elif format == "diacritical":
            # Find first vowel -- where we should put the diacritical mark
            vowels = itertools.chain((c for c in pinyin if c in "aeo"),
                                     (c for c in pinyin if c in "iuv"))
            vowel = pinyin.index(next(vowels)) + 1
            pinyin = pinyin[:vowel] + tonemarks[tone] + pinyin[vowel:]
        else:
            error = "Format must be one of: numerical/diacritical/strip"
            raise ValueError(error)

        yield unicodedata.normalize('NFC', pinyin)
def transform(string, transliterations=None):
    """
    Transform the string to "upside-down" writing.

    Example:

        >>> import upsidedown
        >>> print(upsidedown.transform('Hello World!'))
        ¡pꞁɹoM oꞁꞁǝH

    For languages with diacritics you might want to supply a transliteration to
    work around missing (rendering of) upside-down forms:
        >>> import upsidedown
        >>> print(upsidedown.transform('köln', transliterations={'ö': 'oe'}))
        uꞁǝoʞ
    """
    transliterations = transliterations or TRANSLITERATIONS

    for character in transliterations:
        string = string.replace(character, transliterations[character])

    inputChars = list(string)
    inputChars.reverse()

    output = []
    for character in inputChars:
        if character in _CHARLOOKUP:
            output.append(_CHARLOOKUP[character])
        else:
            charNormalised = unicodedata.normalize("NFD", character)

            for c in charNormalised[:]:
                if c in _CHARLOOKUP:
                    charNormalised = charNormalised.replace(c, _CHARLOOKUP[c])
                elif c in _DIACRITICSLOOKUP:
                    charNormalised = charNormalised.replace(c,
                        _DIACRITICSLOOKUP[c])

            output.append(unicodedata.normalize("NFC", charNormalised))

    return ''.join(output)
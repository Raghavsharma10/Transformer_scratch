def get_kana_info(char):
    """
    Return two things about each character:

    - Its transliterated value (in Roman characters, if it's a kana)
    - A class of characters indicating how it affects the romanization
    """
    try:
        name = unicodedata.name(char)
    except ValueError:
        return char, NOT_KANA

    # The names we're dealing with will probably look like
    # "KATAKANA CHARACTER ZI".
    if (name.startswith('HIRAGANA LETTER') or
        name.startswith('KATAKANA LETTER') or
        name.startswith('KATAKANA-HIRAGANA')):
        names = name.split()
        syllable = str_func(names[-1].lower())

        if name.endswith('SMALL TU'):
            # The small tsu (っ) doubles the following consonant.
            # It'll show up as 't' on its own.
            return 't', SMALL_TSU
        elif names[-1] == 'N':
            return 'n', NN
        elif names[1] == 'PROLONGED':
            # The prolongation marker doubles the previous vowel.
            # It'll show up as '_' on its own.
            return '_', PROLONG
        elif names[-2] == 'SMALL':
            # Small characters tend to modify the sound of the previous
            # kana. If they can't modify anything, they're appended to
            # the letter 'x' instead.
            if syllable.startswith('y'):
                return 'x' + syllable, SMALL_Y
            else:
                return 'x' + syllable, SMALL

        return syllable, KANA
    else:
        if char in ROMAN_PUNCTUATION_TABLE:
            char = ROMAN_PUNCTUATION_TABLE[char]
        return char, NOT_KANA
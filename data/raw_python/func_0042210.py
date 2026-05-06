def kana_romaji_lt(romaji, *kana):
    '''
    Generates a lookup table with the kana characters on the left side
    and their rōmaji equivalents as the values.

    For the consonant-vowel (cv) characters, we'll generate:

       {u'か': ('ka', 'k', 'k', 'ā'),
        u'が': ('ga', 'g', 'g', 'ā'),
        [...]

    Multiple kana character sets can be passed as rest arguments.
    '''
    lt = {}
    for kana_set in kana:
        for n in range(len(romaji)):
            ro = romaji[n]
            ka = kana_set[n]
            lt[ka] = ro

    return lt
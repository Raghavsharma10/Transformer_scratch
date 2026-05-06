def T6(word, rules):
    '''If a VVV-sequence contains a long vowel, insert a syllable boundary
    between it and the third vowel. E.g. [kor.ke.aa], [yh.ti.öön], [ruu.an],
    [mää.yt.te].'''
    offset = 0

    try:
        WORD, rest = tuple(word.split('.', 1))

        for vvv in long_vowel_sequences(rest):
            i = vvv.start(2)
            vvv = vvv.group(2)
            i += (2 if phon.is_long(vvv[:2]) else 1) + offset
            rest = rest[:i] + '.' + rest[i:]
            offset += 1

    except ValueError:
        WORD = word

    for vvv in long_vowel_sequences(WORD):
        i = vvv.start(2) + 2
        WORD = WORD[:i] + '.' + WORD[i:]

    try:
        WORD += '.' + rest

    except UnboundLocalError:
        pass

    rules += ' T6' if word != WORD else ''

    return WORD, rules
def apply_T2(word):
    '''There is a syllable boundary within a VV sequence of two nonidentical
    vowels that are not a genuine diphthong, e.g., [ta.e], [ko.et.taa].'''
    WORD = word
    offset = 0

    for vv in vv_sequences(WORD):
        seq = vv.group(2)

        if not is_diphthong(seq) and not is_long(seq):
            i = vv.start(2) + 1 + offset
            WORD = WORD[:i] + '.' + WORD[i:]
            offset += 1

    RULE = ' T2' if word != WORD else ''

    return WORD, RULE
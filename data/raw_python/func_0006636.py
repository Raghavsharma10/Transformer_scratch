def apply_T12(word):
    '''There is a syllable boundary within a VV sequence of two nonidentical
    vowels that are not a genuine diphthong, e.g., [ta.e], [ko.et.taa].'''
    WORD = word
    offset = 0

    for vv in new_vv(WORD):
        # import pdb; pdb.set_trace()
        seq = vv.group(1)

        if not is_diphthong(seq) and not is_long(seq):
            i = vv.start(1) + 1 + offset
            WORD = WORD[:i] + '.' + WORD[i:]
            offset += 1

    RULE = ' T2' if word != WORD else ''

    return WORD, RULE
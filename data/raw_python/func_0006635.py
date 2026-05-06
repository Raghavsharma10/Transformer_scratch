def apply_T11(word):
    '''If a VVV sequence contains a /u, y/-final diphthong and the third vowel
    is /i/, there is a syllable boundary between the diphthong and /i/.'''
    WORD = word
    offset = 0

    for vvv in t11_vvv_sequences(WORD):
        # i = vvv.start(1) + (1 if vvv.group(1).startswith('i') else 2) + offset
        i = vvv.start(1) + (1 if vvv.group(1)[-1] in 'uy' else 2) + offset
        WORD = WORD[:i] + '.' + WORD[i:]
        offset += 1

    RULE = ' T11' if word != WORD else ''

    return WORD, RULE
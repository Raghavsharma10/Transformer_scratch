def apply_T5(word):  # BROKEN
    '''If a (V)VVV-sequence contains a VV-sequence that could be an /i/-final
    diphthong, there is a syllable boundary between it and the third vowel,
    e.g., [raa.ois.sa], [huo.uim.me], [la.eis.sa], [sel.vi.äi.si], [tai.an],
    [säi.e], [oi.om.me].'''
    T5 = ''
    WORD = word.split('.')

    for i, v in enumerate(WORD):
        if contains_VVV(v) and any(i for i in i_DIPHTHONGS if i in v):
            I = v.rfind('i') - 1 or 2
            I = I + 2 if is_consonant(v[I - 1]) else I
            WORD[i] = v[:I] + '.' + v[I:]
            T5 = ' T5'

    word = '.'.join(WORD)

    return word, T5
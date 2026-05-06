def apply_T2(word):
    '''There is a syllable boundary within a sequence VV of two nonidentical
    vowels that are not a genuine diphthong, e.g., [ta.e], [ko.et.taa].'''
    T2 = ''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if not contains_diphthong(v):
            VV = contains_VV(v)

            if VV:
                I = v.find(VV) + 1
                WORD[i] = v[:I] + '.' + v[I:]
                T2 = ' T2'

    word = '.'.join(WORD)

    return word, T2
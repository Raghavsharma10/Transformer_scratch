def apply_T6(word):
    '''If a VVV-sequence contains a long vowel, there is a syllable boundary
    between it and the third vowel, e.g. [kor.ke.aa], [yh.ti.öön], [ruu.an],
    [mää.yt.te].'''
    T6 = ''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if contains_VVV(v):
            VV = [v.find(j) for j in LONG_VOWELS if v.find(j) > 0]

            if VV:
                I = VV[0]
                T6 = ' T6'

                if I + 2 == len(v) or is_vowel(v[I + 2]):
                    WORD[i] = v[:I + 2] + '.' + v[I + 2:]  # TODO

                else:
                    WORD[i] = v[:I] + '.' + v[I:]

    word = '.'.join(WORD)
    word = word.strip('.')  # TODO

    return word, T6
def apply_T7(word):
    '''If a VVV-sequence does not contain a potential /i/-final diphthong,
    there is a syllable boundary between the second and third vowels, e.g.
    [kau.an], [leu.an], [kiu.as].'''
    T7 = ''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        if contains_VVV(v):

            for I, V in enumerate(v[::-1]):

                if is_vowel(V):
                    WORD[i] = v[:I] + '.' + v[I:]
                    T7 = ' T7'

    word = '.'.join(WORD)

    return word, T7
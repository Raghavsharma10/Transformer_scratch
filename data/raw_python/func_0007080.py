def apply_T4(word):
    '''An agglutination diphthong that ends in /u, y/ usually contains a
    syllable boundary when -C# or -CCV follow, e.g., [lau.ka.us],
    [va.ka.ut.taa].'''
    T4 = ''
    WORD = word.split('.')

    for i, v in enumerate(WORD):

        # i % 2 != 0 prevents this rule from applying to first, third, etc.
        # syllables, which receive stress (WSP)
        if is_consonant(v[-1]) and i % 2 != 0:

            if i + 1 == len(WORD) or is_consonant(WORD[i + 1][0]):

                if contains_Vu_diphthong(v):
                    I = v.rfind('u')
                    WORD[i] = v[:I] + '.' + v[I:]
                    T4 = ' T4'

                elif contains_Vy_diphthong(v):
                    I = v.rfind('y')
                    WORD[i] = v[:I] + '.' + v[I:]
                    T4 = ' T4'

    word = '.'.join(WORD)

    return word, T4
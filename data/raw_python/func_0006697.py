def apply_T4(word):
    '''An agglutination diphthong that ends in /u, y/ optionally contains a
    syllable boundary when -C# or -CCV follow, e.g., [lau.ka.us],
    [va.ka.ut.taa].'''
    WORD = word.split('.')
    PARTS = [[] for part in range(len(WORD))]

    for i, v in enumerate(WORD):

        # i % 2 != 0 prevents this rule from applying to first, third, etc.
        # syllables, which receive stress (WSP)
        if is_consonant(v[-1]) and i % 2 != 0:
            if i + 1 == len(WORD) or is_consonant(WORD[i + 1][0]):
                vv = u_y_final_diphthongs(v)

                if vv:
                    I = vv.start(1) + 1
                    PARTS[i].append(v[:I] + '.' + v[I:])

        # include original form (non-application of rule)
        PARTS[i].append(v)

    WORDS = [w for w in product(*PARTS)]

    for WORD in WORDS:
        WORD = '.'.join(WORD)
        RULE = ' T4' if word != WORD else ''

        yield WORD, RULE
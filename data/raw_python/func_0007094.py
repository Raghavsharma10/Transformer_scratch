def apply_T4(word):  # OPTIMIZE
    '''An agglutination diphthong that ends in /u, y/ usually contains a
    syllable boundary when -C# or -CCV follow, e.g., [lau.ka.us],
    [va.ka.ut.taa].'''
    WORD = _split_consonants_and_vowels(word)

    for k, v in WORD.iteritems():

        if len(v) == 2 and v.endswith(('u', 'y')):

            if WORD.get(k + 2, 0):

                if not WORD.get(k + 3, 0):
                    if len(WORD[k + 2]) == 1 and is_consonant(WORD[k + 2]):
                        WORD[k] = v[0] + '.' + v[1]

                elif len(WORD[k + 1]) == 1 and WORD.get(k + 3, 0):
                    if is_consonant(WORD[k + 3][0]):
                        WORD[k] = v[0] + '.' + v[1]

                elif len(WORD[k + 2]) == 2:
                    WORD[k] = v[0] + '.' + v[1]

    word = _compile_dict_into_word(WORD)

    return word
def apply_T7(word):
    '''If a VVV-sequence does not contain a potential /i/-final diphthong,
    there is a syllable boundary between the second and third vowels, e.g.
    [kau.an], [leu.an], [kiu.as].'''
    WORD = _split_consonants_and_vowels(word)

    for k, v in WORD.iteritems():

        if len(v) == 3 and is_vowel(v[0]):
            WORD[k] = v[:2] + '.' + v[2:]

    word = _compile_dict_into_word(WORD)

    return word
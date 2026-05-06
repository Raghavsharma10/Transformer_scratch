def apply_T2(word):
    '''There is a syllable boundary within a sequence VV of two nonidentical
    that are not a genuine diphthong, e.g., [ta.e], [ko.et.taa].'''
    WORD = _split_consonants_and_vowels(word)

    for k, v in WORD.iteritems():

        if is_diphthong(v):
            continue

        if len(v) == 2 and is_vowel(v[0]):

            if v[0] != v[1]:
                WORD[k] = v[0] + '.' + v[1]

    word = _compile_dict_into_word(WORD)

    return word
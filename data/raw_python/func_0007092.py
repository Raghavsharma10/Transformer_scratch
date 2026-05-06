def apply_T1(word):
    '''There is a syllable boundary in front of every CV-sequence.'''
    WORD = _split_consonants_and_vowels(word)

    for k, v in WORD.iteritems():

        if k == 1 and is_consonantal_onset(v):
            WORD[k] = '.' + v

        elif is_consonant(v[0]) and WORD.get(k + 1, 0):
            WORD[k] = v[:-1] + '.' + v[-1]

    word = _compile_dict_into_word(WORD)

    return word
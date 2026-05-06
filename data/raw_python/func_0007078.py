def apply_T1(word):
    '''There is a syllable boundary in front of every CV-sequence.'''
    T1 = ' T1'
    WORD = _split_consonants_and_vowels(word)
    CONTINUE_VV = 0
    CONTINUE_VVV = 0

    for i, v in enumerate(WORD):

        if i == 0 and is_consonant(v[0][0]):
            continue

        elif is_consonant(v[0]) and i + 1 != len(WORD):
            WORD[i] = v[:-1] + '.' + v[-1]

        elif is_vowel(v[0]):

            if len(v) > 2:
                CONTINUE_VVV += 1

            elif len(v) > 1:
                CONTINUE_VV += 1

    word = ''.join(WORD)

    return word, CONTINUE_VV, CONTINUE_VVV, T1
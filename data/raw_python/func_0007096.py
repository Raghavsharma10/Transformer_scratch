def apply_T6(word):
    '''If a VVV-sequence contains a long vowel, there is a syllable boundary
    between it and the third vowel, e.g. [kor.ke.aa], [yh.ti.öön], [ruu.an],
    [mää.yt.te].'''
    WORD = _split_consonants_and_vowels(word)

    for k, v in WORD.iteritems():

        if len(v) == 3 and is_vowel(v[0]):
            vv = [v.find(i) for i in LONG_VOWELS if v.find(i) > 0]

            if any(vv):
                vv = vv[0]

                if vv == v[0]:
                    WORD[k] = v[:2] + '.' + v[2:]

                else:
                    WORD[k] = v[:vv] + '.' + v[vv:]

    word = _compile_dict_into_word(WORD)

    return word
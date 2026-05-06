def apply_T5(word):
    '''If a (V)VVV-sequence contains a VV-sequence that could be an /i/-final
    diphthong, there is a syllable boundary between it and the third vowel,
    e.g., [raa.ois.sa], [huo.uim.me], [la.eis.sa], [sel.vi.äi.si], [tai.an],
    [säi.e], [oi.om.me].'''
    WORD = _split_consonants_and_vowels(word)

    for k, v in WORD.iteritems():

        if len(v) >= 3 and is_vowel(v[0]):
            vv = [v.find(i) for i in i_DIPHTHONGS if v.find(i) > 0]

            if any(vv):
                vv = vv[0]

                if vv == v[0]:
                    WORD[k] = v[:2] + '.' + v[2:]

                else:
                    WORD[k] = v[:vv] + '.' + v[vv:]

    word = _compile_dict_into_word(WORD)

    return word
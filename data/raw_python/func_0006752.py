def apply_T1(word):
    '''There is a syllable boundary in front of every CV-sequence.'''
    # split consonants and vowels: 'balloon' -> ['b', 'a', 'll', 'oo', 'n']
    WORD = [w for w in re.split('([ieAyOauo]+)', word) if w]
    count = 0

    for i, v in enumerate(WORD):

        if i == 0 and is_consonant(v[0]):
            continue

        elif is_consonant(v[0]) and i + 1 != len(WORD):
            if is_cluster(v):  # WSP
                if count % 2 == 0:
                    WORD[i] = v[0] + '.' + v[1:]  # CC > C.C, CCC > C.CC

                else:
                    WORD[i] = '.' + v  # CC > .CC, CCC > .CCC

            # elif is_sonorant(v[0]) and is_cluster(v[1:]):  # NEW
            #     if count % 2 == 0:
            #         WORD[i] = v[0:2] + '.' + v[2:]

            #     else:
            #         WORD[i] = v[0] + '.' + v[1:]

            else:
                WORD[i] = v[:-1] + '.' + v[-1]  # CC > C.C, CCC > CC.C

            count += 1

    WORD = ''.join(WORD)
    RULE = ' T1' if word != WORD else ''

    return WORD, RULE
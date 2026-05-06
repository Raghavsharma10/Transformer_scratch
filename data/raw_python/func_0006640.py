def T1(word):
    '''Insert a syllable boundary in front of every CV sequence.'''
    # split consonants and vowels: 'balloon' -> ['b', 'a', 'll', 'oo', 'n']
    WORD = [i for i in re.split(r'([ieaouäöy]+)', word, flags=FLAGS) if i]

    # keep track of which sub-rules are applying
    sub_rules = set()

    # a count divisible by 2 indicates an even syllable
    count = 1

    for i, v in enumerate(WORD):

        # T1B
        # If there is a consonant cluster word-initially, the entire cluster
        # forms the onset of the first syllable:
        # CCV > #CCV
        if i == 0 and phon.is_consonant(v[0]):
            sub_rules.add('b')

        elif phon.is_consonant(v[0]):
            count += 1

            # True if the current syllable is unstressed, else False
            unstressed = count % 2 == 0

            # T1C
            # If there is a consonant cluster word-finally, the entire cluster
            # forms the coda of the final syllable:
            # VCC# > VCC#
            if i + 1 == len(WORD):
                sub_rules.add('c')

            # T1D
            # If there is a bare "Finnish" consonant cluster word-medially and
            # the previous syllable receives stress, the first consonant of the
            # cluster forms the coda of the previous syllable (to create a
            # heavy syllable); otherwise, the whole cluster forms the onset of
            # the current syllable (this is the /kr/ rule):
            # 'VCCV > 'VC.CV,  VCCV > V.CCV
            elif phon.is_cluster(v):
                sub_rules.add('d')
                WORD[i] = v[0] + '.' + v[1:] if unstressed else '.' + v

            elif phon.is_cluster(v[1:]):

                # T1E (optional)
                # If there is a word-medial "Finnish" consonant cluster that is
                # preceded by a sonorant consonant, if the previous syllable
                # receives stress, the sonorant consonant and the first
                # consonant of the cluster form the coda of the previous
                # syllable, and the remainder of the cluster forms the onset of
                # the current syllable:
                # 'VlCC > VlC.C
                if phon.is_sonorant(v[0]) and unstressed:
                    sub_rules.add('e')
                    WORD[i] = v[:2] + '.' + v[2:]

                # T1F
                # If there is a word-medial "Finnish" cluster that follows a
                # consonant, that first consonant forms the coda of the
                # previous syllable, and the cluster forms the onset of the
                # current syllable:
                # VCkr > VC.kr
                else:
                    sub_rules.add('f')
                    WORD[i] = v[0] + '.' + v[1:]

            # T1A
            # There is a syllable boundary in front of every CV sequence:
            # VCV > V.CV, CCV > C.CV
            else:
                WORD[i] = v[:-1] + '.' + v[-1]
                sub_rules.add('a')

    WORD = ''.join(WORD)
    rules = '' if word == WORD else ' T1'  # + ''.join(sub_rules)  # TODO: sort

    return WORD, rules
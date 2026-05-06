def _syllabify_simplex(word):
    '''Syllabify the given word.'''
    word, rules = apply_T1(word)

    if re.search(r'[^ieAyOauo]*([ieAyOauo]{2})[^ieAyOauo]*', word):
        word, T2 = apply_T2(word)
        word, T8 = apply_T8(word)
        word, T9 = apply_T9(word)
        rules += T2 + T8 + T9

        # T4 produces variation
        syllabifications = list(apply_T4(word))

    else:
        syllabifications = [(word, ''), ]

    for word, rule in syllabifications:
        RULES = rules + rule

        if re.search(r'[ieAyOauo]{3}', word):
            word, T6 = apply_T6(word)
            word, T5 = apply_T5(word)
            word, T10 = apply_T10(word)
            word, T7 = apply_T7(word)
            word, T2 = apply_T2(word)
            RULES += T5 + T6 + T10 + T7 + T2

        RULES = RULES or ' T0'  # T0 means no rules have applied

        yield word, RULES
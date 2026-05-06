def _syllabify(word):
    '''Syllabify the given word.'''
    word = replace_umlauts(word)
    word, CONTINUE_VV, CONTINUE_VVV, applied_rules = apply_T1(word)

    if CONTINUE_VV:
        word, T2 = apply_T2(word)
        word, T4 = apply_T4(word)
        applied_rules += T2 + T4

    if CONTINUE_VVV:
        word, T5 = apply_T5(word)
        word, T6 = apply_T6(word)
        word, T7 = apply_T7(word)
        applied_rules += T5 + T6 + T7

    word = replace_umlauts(word, put_back=True)

    return word, applied_rules
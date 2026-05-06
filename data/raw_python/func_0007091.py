def syllabify(word):
    '''Syllabify the given word.'''

    word = replace_umlauts(word)

    word = apply_T1(word)
    word = apply_T2(word)
    word = apply_T4(word)
    word = apply_T5(word)
    word = apply_T6(word)
    word = apply_T7(word)

    word = replace_umlauts(word, put_back=True)[1:]  # FENCEPOST

    return word
def T8(word, rules):
    '''Join /ie/, /uo/, or /yö/ sequences in syllables that take primary
    stress.'''
    WORD = word

    try:
        vv = tail_diphthongs(WORD)
        i = vv.start(1) + 1
        WORD = WORD[:i] + word[i + 1:]

    except AttributeError:
        pass

    rules += ' T8' if word != WORD else ''

    return WORD, rules
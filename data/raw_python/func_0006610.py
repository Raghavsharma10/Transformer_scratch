def apply_T8(word):
    '''Split /ie/ sequences in syllables that do not take primary stress.'''
    WORD = word
    offset = 0

    for ie in ie_sequences(WORD):
        i = ie.start(1) + 1 + offset
        WORD = WORD[:i] + '.' + WORD[i:]
        offset += 1

    RULE = ' T8' if word != WORD else ''

    return WORD, RULE
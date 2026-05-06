def apply_T8(word):
    '''Split /ie/, /uo/, or /yö/ sequences in syllables that do not take
    primary stress.'''
    WORD = word
    offset = 0

    for vv in tail_diphthongs(WORD):
        i = vv.start(1) + 1 + offset
        WORD = WORD[:i] + '.' + WORD[i:]
        offset += 1

    RULE = ' T8' if word != WORD else ''

    return WORD, RULE
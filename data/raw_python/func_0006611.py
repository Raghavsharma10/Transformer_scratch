def apply_T9(word):
    '''Split /iu/ sequences that do not appear in the first, second, or final
    syllables.'''
    WORD = word
    index = 0
    offset = 0

    for iu in iu_sequences(WORD):
        if iu.start(1) != index:
            i = iu.start(1) + 1 + offset
            WORD = WORD[:i] + '.' + WORD[i:]
            index = iu.start(1)
            offset += 1

    RULE = ' T9' if word != WORD else ''

    return WORD, RULE
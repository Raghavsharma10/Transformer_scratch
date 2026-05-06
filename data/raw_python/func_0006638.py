def apply_T9(word):
    '''Split /iu/ sequences that do not appear in the first or second
    syllables. Split /iu/ sequences in the final syllable iff the final
    syllable would receive stress.'''
    WORD = word
    index = 0
    offset = 0

    for iu in iu_sequences(WORD):
        if iu.start(1) != index:
            i = iu.start(1) + 1 + offset
            WORD = WORD[:i] + '.' + WORD[i:]
            index = iu.start(1)
            offset += 1

    # split any /iu/ sequence in the final syllable iff the final syllable
    # would receive stress -- to capture extrametricality
    if WORD.count('.') % 2 == 0:
        iu = iu_sequences(WORD, word_final=True)

        if iu:
            i = iu.start(1) + 1
            WORD = WORD[:i] + '.' + WORD[i:]

    RULE = ' T9' if word != WORD else ''

    return WORD, RULE
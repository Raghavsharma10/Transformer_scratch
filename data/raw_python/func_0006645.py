def T11(word, rules):
    '''If a VVV sequence contains a /u,y/-final diphthong, insert a syllable
    boundary between the diphthong and the third vowel.'''
    WORD = word
    offset = 0

    for vvv in precedence_sequences(WORD):
        i = vvv.start(1) + (1 if vvv.group(1)[-1] in 'uyUY' else 2) + offset
        WORD = WORD[:i] + '.' + WORD[i:]
        offset += 1

    rules += ' T11' if word != WORD else ''

    return WORD, rules
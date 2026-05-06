def T2(word, rules):
    '''Split any VV sequence that is not a genuine diphthong or long vowel.
    E.g., [ta.e], [ko.et.taa]. This rule can apply within VVV+ sequences.'''
    WORD = word
    offset = 0

    for vv in vv_sequences(WORD):
        seq = vv.group(1)

        if not phon.is_diphthong(seq) and not phon.is_long(seq):
            i = vv.start(1) + 1 + offset
            WORD = WORD[:i] + '.' + WORD[i:]
            offset += 1

    rules += ' T2' if word != WORD else ''

    return WORD, rules
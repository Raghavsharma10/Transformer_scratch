def apply_T10(word):
    '''Any /iou/ sequence contains a syllable boundary between the first and
    second vowel.'''
    WORD = word
    offset = 0

    for iou in iou_sequences(WORD):
        i = iou.start(1) + 1 + offset
        WORD = WORD[:i] + '.' + WORD[i:]
        offset += 1

    RULE = ' T10' if word != WORD else ''

    return WORD, RULE
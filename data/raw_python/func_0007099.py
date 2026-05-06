def wsp(word):
    '''Return the number of unstressed superheavy syllables.'''
    violations = 0
    unstressed = []

    for w in extract_words(word):
        unstressed += w.split('.')[1::2]  # even syllables

        # include extrametrical odd syllables as potential WSP violations
        if w.count('.') % 2 == 0:
            unstressed += [w.rsplit('.', 1)[-1], ]

    # SHSP
    for syll in unstressed:
        if re.search(r'[ieaouäöy]{2}[^$ieaouäöy]+', syll, flags=FLAGS):
            violations += 1

    # # WSP (CVV = heavy)
    # for syll in unstressed:
    #     if re.search(
    #             ur'[ieaouäöy]{2}|[ieaouäöy]+[^ieaouäöy]+',
    #             syll, flags=re.I | re.U):
    #         violations += 1

    return violations
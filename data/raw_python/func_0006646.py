def pk_prom(word):
    '''Return the number of stressed light syllables.'''
    violations = 0
    stressed = []

    for w in extract_words(word):
        stressed += w.split('.')[2:-1:2]  # odd syllables, excl. word-initial

    # (CVV = light)
    for syll in stressed:
        if phon.is_vowel(syll[-1]):
            violations += 1

    # # (CVV = heavy)
    # for syll in stressed:
    #     if re.search(
    #             ur'^[^ieaouäöy]*[ieaouäöy]{1}$',  syll, flags=re.I | re.U):
    #         violations += 1

    return violations
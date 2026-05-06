def syllabify(word):
    '''Syllabify the given word, whether simplex or complex.'''
    compound = not word.isalpha()
    syllabify = _syllabify_complex if compound else _syllabify_simplex
    syllabifications = list(syllabify(word))

    # if variation, order variants from most preferred to least preferred
    if len(syllabifications) > 1:
        syllabifications = rank(syllabifications)

    for word, rules in syllabifications:
        yield _post_process(word, rules)
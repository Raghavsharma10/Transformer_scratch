def syllabify(word):
    '''Syllabify the given word, whether simplex or complex.'''
    compound = bool(re.search(r'(-| |=)', word))
    syllabify = _syllabify_compound if compound else _syllabify
    syllabifications = list(syllabify(word))

    for syll, rules in syllabifications:
        yield syll, rules

    n = 16 - len(syllabifications)

    # yield empty syllabifications and rules
    for i in range(n):
        yield '', ''
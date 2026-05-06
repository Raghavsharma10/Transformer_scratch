def syllabify(word, compound=None):
    '''Syllabify the given word, whether simplex or complex.'''
    if compound is None:
        compound = bool(re.search(r'(-| |=)', word))

    syllabify = _syllabify_compound if compound else _syllabify
    syll, rules = syllabify(word)

    yield syll, rules

    n = 7

    if 'T4' in rules:
        yield syllabify(word, T4=False)
        n -= 1

    if 'e' in rules:
        yield syllabify(word, T1E=False)
        n -= 1

    if 'e' in rules and 'T4' in rules:
        yield syllabify(word, T4=False, T1E=False)
        n -= 1

    # yield empty syllabifications and rules
    for i in range(n):
        yield '', ''
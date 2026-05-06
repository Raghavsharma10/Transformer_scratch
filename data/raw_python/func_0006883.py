def syllabify(word):
    '''Syllabify the given word, whether simplex or complex.'''
    word = split(word)  # detect any non-delimited compounds
    compound = True if re.search(r'-| |\.', word) else False
    syllabify = _syllabify_compound if compound else _syllabify
    syll, rules = syllabify(word)

    yield syll, rules

    n = 3

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
    for n in range(7):
        yield '', ''
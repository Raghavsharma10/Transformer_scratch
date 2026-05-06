def syllabify(word):
    '''Syllabify the given word, whether simplex or complex.'''
    compound = bool(re.search(r'(-| |=)', word))
    syllabify = _syllabify_compound if compound else _syllabify_simplex
    syllabifications = list(syllabify(word))

    for word, rules in rank(syllabifications):
        # post-process
        word = str(replace_umlauts(word, put_back=True))
        rules = rules[1:]

        yield word, rules
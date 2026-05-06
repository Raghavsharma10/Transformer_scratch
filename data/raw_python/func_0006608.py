def pk_prom(word):
    '''Return the number of stressed light syllables.'''
    LIGHT = r'[ieaAoO]{1}[\.]*(u|y)(\.|$)'

    # # if the word is not monosyllabic, lop off the final syllable, which is
    # # extrametrical
    # if '.' in word:
    #     word = word[:word.rindex('.')]

    # gather the indices of syllable boundaries
    delimiters = [0, ] + [i for i, char in enumerate(word) if char == '.']

    if len(delimiters) % 2 != 0:
        delimiters.append(len(word))

    stressed = []

    # gather the indices of stressed positions
    for i, d in enumerate(delimiters):
        if i % 2 == 0:
            stressed.extend(range(d + 1, delimiters[i + 1]))

    # find the number of stressed light syllables
    heavies = re.finditer(LIGHT, word)
    violations = sum(1 for m in heavies if m.start(1) in stressed)

    return violations
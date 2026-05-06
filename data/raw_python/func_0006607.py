def wsp(word):
    '''Return the number of unstressed heavy syllables.'''
    HEAVY = r'[ieaAoO]{1}[\.]*(u|y)[^ieaAoO]+(\.|$)'

    # # if the word is not monosyllabic, lop off the final syllable, which is
    # # extrametrical
    # if '.' in word:
    #     word = word[:word.rindex('.')]

    # gather the indices of syllable boundaries
    delimiters = [i for i, char in enumerate(word) if char == '.']

    if len(delimiters) % 2 != 0:
        delimiters.append(len(word))

    unstressed = []

    # gather the indices of unstressed positions
    for i, d in enumerate(delimiters):
        if i % 2 == 0:
            unstressed.extend(range(d + 1, delimiters[i + 1]))

    # find the number of unstressed heavy syllables
    heavies = re.finditer(HEAVY, word)
    violations = sum(1 for m in heavies if m.start(0) in unstressed)

    return violations
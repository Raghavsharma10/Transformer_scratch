def harmonic(word):
    '''Return True if the word's vowels agree in frontness/backness.'''
    depth = {'ä': 0, 'ö': 0, 'y': 0, 'a': 1, 'o': 1, 'u': 1}
    vowels = filter(lambda ch: is_front(ch) or is_back(ch), word)
    depths = (depth[x.lower()] for x in vowels)

    return len(set(depths)) < 2
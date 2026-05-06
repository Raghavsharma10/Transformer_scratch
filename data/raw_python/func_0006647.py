def rank(syllabifications):
    '''Rank syllabifications.'''

    # def key(s):
    #     word = s[0]
    #     w = wsp(word)
    #     p = pk_prom(word)
    #     n = nuc(word)
    #     t = w + p + n
    #     print('%s\twsp: %s\tpk: %s\tnuc: %s\ttotal: %s' % (word, w, p, n, t))

    #     return w + p + n

    # syllabifications.sort(key=key)

    syllabifications.sort(key=lambda s: wsp(s[0]) + pk_prom(s[0]) + nuc(s[0]))

    return syllabifications
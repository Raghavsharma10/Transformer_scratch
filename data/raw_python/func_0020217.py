def tolerant_metaphone_processor(words):
    '''Double metaphone word processor slightly modified so that when no
words are returned by the algorithm, the original word is returned.'''
    for word in words:
        r = 0
        for w in double_metaphone(word):
            if w:
                w = w.strip()
                if w:
                    r += 1
                    yield w
        if not r:
            yield word
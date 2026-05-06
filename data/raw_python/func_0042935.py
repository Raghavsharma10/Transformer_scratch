def filter_ascii(lst):
    '''
    removes words with accent chars etc.
    (most accented words in the english lookup exist in the same table unaccented.)
    '''
    return [word for word in lst if all(ord(c) < 128 for c in word)]
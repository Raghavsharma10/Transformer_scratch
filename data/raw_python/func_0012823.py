def replaceint(fname, replacewith='%s'):
    """replace int in lst"""
    words = fname.split()
    for i, word in enumerate(words):
        try:
            word = int(word)
            words[i] = replacewith
        except ValueError:
            pass
    return ' '.join(words)
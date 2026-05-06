def load_words(wordlist=None):
    """ Loads and caches the profanity word list. Input file (if provided)
    should be a flat text file with one profanity entry per line.

    """
    global words
    if not wordlist:
        # no wordlist was provided, load the wordlist from the local store
        filename = get_data('wordlist.txt')
        f = open(filename)
        wordlist = f.readlines()
        wordlist = [w.strip() for w in wordlist if w]
    words = wordlist
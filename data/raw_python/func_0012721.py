def string_pieces(s, maxlen=1024):
    """
    Takes a (unicode) string and yields pieces of it that are at most `maxlen`
    characters, trying to break it at punctuation/whitespace. This is an
    important step before using a tokenizer with a maximum buffer size.
    """
    if not s:
        return
    i = 0
    while True:
        j = i + maxlen
        if j >= len(s):
            yield s[i:]
            return
        # Using "j - 1" keeps boundary characters with the left chunk
        while unicodedata.category(s[j - 1]) not in BOUNDARY_CATEGORIES:
            j -= 1
            if j == i:
                # No boundary available; oh well.
                j = i + maxlen
                break
        yield s[i:j]
        i = j
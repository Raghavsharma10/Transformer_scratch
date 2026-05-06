def language(s):
    """ Returns a (language, confidence)-tuple for the given string.
    """
    s = decode_utf8(s)
    s = set(w.strip(PUNCTUATION) for w in s.replace("'", "' ").split())
    n = float(len(s) or 1)
    p = {}
    for xx in LANGUAGES:
        lexicon = _module(xx).__dict__["lexicon"]
        p[xx] = sum(1 for w in s if w in lexicon) / n
    return max(p.items(), key=lambda kv: (kv[1], int(kv[0] == "en")))
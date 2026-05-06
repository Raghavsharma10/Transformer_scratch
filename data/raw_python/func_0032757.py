def parseSearchTerm(term):
    """
    Turn a string search query into a two-tuple of a search term and a
    dictionary of search keywords.
    """
    terms = []
    keywords = {}
    for word in term.split():
        if word.count(':') == 1:
            k, v = word.split(u':')
            if k and v:
                keywords[k] = v
            elif k or v:
                terms.append(k or v)
        else:
            terms.append(word)
    term = u' '.join(terms)
    if keywords:
        return term, keywords
    return term, None
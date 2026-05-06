def singularize(word, pos=NOUN, gender=MALE, role=SUBJECT, custom={}):
    """ Returns the singular of a given word.
        The inflection is based on probability rather than gender and role.
    """
    w = word.lower().capitalize()
    if word in custom:
        return custom[word]
    if word in singular:
        return singular[word]
    if pos == NOUN:
        for a, b in singular_inflections:
            if w.endswith(a):
                return w[:-len(a)] + b
        # Default rule: strip known plural suffixes (baseline = 51%).
        for suffix in ("nen", "en", "n", "e", "er", "s"):
            if w.endswith(suffix):
                w = w[:-len(suffix)]
                break
        # Corrections (these add about 1% accuracy):
        if w.endswith(("rr", "rv", "nz")):
            return w + "e"
        return w
    return w
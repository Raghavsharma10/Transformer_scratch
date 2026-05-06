def pluralize(word, pos=NOUN, gender=MALE, role=SUBJECT, custom={}):
    """ Returns the plural of a given word.
        The inflection is based on probability rather than gender and role.
    """
    w = word.lower().capitalize()
    if word in custom:
        return custom[word]
    if pos == NOUN:
        for a, b in plural_inflections:
            if w.endswith(a):
                return w[:-len(a)] + b
        # Default rules (baseline = 69%).
        if w.startswith("ge"):
            return w
        if w.endswith("gie"):
            return w
        if w.endswith("e"):
            return w + "n"
        if w.endswith("ien"):
            return w[:-2] + "um"
        if w.endswith(("au", "ein", "eit", "er", "en", "el", "chen", "mus", u"tät", "tik", "tum", "u")):
            return w
        if w.endswith(("ant", "ei", "enz", "ion", "ist", "or", "schaft", "tur", "ung")):
            return w + "en"
        if w.endswith("in"):
            return w + "nen"
        if w.endswith("nis"):
            return w + "se"
        if w.endswith(("eld", "ild", "ind")):
            return w + "er"
        if w.endswith("o"):
            return w + "s"
        if w.endswith("a"):
            return w[:-1] + "en"
        # Inflect common umlaut vowels: Kopf => Köpfe.
        if w.endswith(("all", "and", "ang", "ank", "atz", "auf", "ock", "opf", "uch", "uss")):
            umlaut = w[-3]
            umlaut = umlaut.replace("a", u"ä")
            umlaut = umlaut.replace("o", u"ö")
            umlaut = umlaut.replace("u", u"ü")
            return w[:-3] + umlaut + w[-2:] + "e"
        for a, b in (
          ("ag",  u"äge"), 
          ("ann", u"änner"), 
          ("aum", u"äume"), 
          ("aus", u"äuser"), 
          ("zug", u"züge")):
            if w.endswith(a):
                return w[:-len(a)] + b
        return w + "e"
    return w
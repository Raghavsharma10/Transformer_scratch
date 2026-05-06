def indefinite_article(word, gender=MALE, role=SUBJECT):
    """ Returns the indefinite article (ein) for a given word.
    """
    return article_indefinite.get((gender[:1].lower(), role[:3].lower()))
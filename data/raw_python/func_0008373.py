def definite_article(word, gender=MALE, role=SUBJECT):
    """ Returns the definite article (der/die/das/die) for a given word.
    """
    return article_definite.get((gender[:1].lower(), role[:3].lower()))
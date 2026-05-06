def referenced(word, article=INDEFINITE, gender=MALE, role=SUBJECT):
    """ Returns a string with the article + the word.
    """
    return "%s %s" % (_article(word, article, gender, role), word)
def article(word, function=INDEFINITE, gender=MALE, role=SUBJECT):
    """ Returns the indefinite (ein) or definite (der/die/das/die) article for the given word.
    """
    return function == DEFINITE \
       and definite_article(word, gender, role) \
        or indefinite_article(word, gender, role)
def setTerms(variant):
    """
    Sets the default terms of the Dignities
    table.

    """
    global TERMS
    if variant == EGYPTIAN_TERMS:
        TERMS = tables.EGYPTIAN_TERMS
    elif variant == TETRABIBLOS_TERMS:
        TERMS = tables.TETRABIBLOS_TERMS
    elif variant == LILLY_TERMS:
        TERMS = tables.LILLY_TERMS
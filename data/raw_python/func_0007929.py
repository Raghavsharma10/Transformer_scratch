def setFaces(variant):
    """
    Sets the default faces variant

    """
    global FACES
    if variant == CHALDEAN_FACES:
        FACES = tables.CHALDEAN_FACES
    else:
        FACES = tables.TRIPLICITY_FACES
def face(sign, lon):
    """ Returns the face for a sign and longitude. """
    faces = FACES[sign]
    if lon < 10:
        return faces[0]
    elif lon < 20:
        return faces[1]
    else:
        return faces[2]
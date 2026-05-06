def tomindec(origin):
    """
    Convert [+/-]DDD.DDDDD to a tuple (degrees, minutes)
    """

    origin = float(origin)
    degrees = int(origin)
    minutes = (origin % 1) * 60

    return degrees, minutes
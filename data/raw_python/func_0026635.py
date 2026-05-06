def todms(origin):
    """
    Convert [+/-]DDD.DDDDD to a tuple (degrees, minutes, seconds)
    """

    degrees, minutes = tomindec(origin)
    seconds = (minutes % 1) * 60

    return degrees, int(minutes), seconds
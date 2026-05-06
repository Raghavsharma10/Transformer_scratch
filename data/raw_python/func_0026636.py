def todmsstr(origin):
    """
    Convert [+/-]DDD.DDDDD to [+/-]DDD°MMM'DDD.DDDDD"
    """

    degrees, minutes, seconds = todms(origin)
    return u'%d°%d\'%f"' % (degrees, minutes, seconds)
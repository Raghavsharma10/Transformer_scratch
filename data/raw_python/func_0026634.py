def tomindecstr(origin):
    """
    Convert [+/-]DDD.DDDDD to [+/-]DDD°MMM.MMMM'
    """

    degrees, minutes = tomindec(origin)
    return u'%d°%f\'' % (degrees, minutes)
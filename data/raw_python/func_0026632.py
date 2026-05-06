def todegdec(origin):
    """
    Convert from [+/-]DDD°MMM'SSS.SSSS" or [+/-]DDD°MMM.MMMM' to [+/-]DDD.DDDDD
    """

    # if the input is already a float (or can be converted to float)
    try:
        return float(origin)
    except ValueError:
        pass

    # DMS format
    m = dms_re.search(origin)
    if m:
        degrees = int(m.group('degrees'))
        minutes = float(m.group('minutes'))
        seconds = float(m.group('seconds'))

        return degrees + minutes / 60 + seconds / 3600

    # Degree + Minutes format
    m = mindec_re.search(origin)
    if m:
        degrees = int(m.group('degrees'))
        minutes = float(m.group('minutes'))

        return degrees + minutes / 60
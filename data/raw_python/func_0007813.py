def nextStation(ID, date):
    """ Returns the aproximate date of the next station. """
    jd = eph.nextStation(ID, date.jd)
    return Datetime.fromJD(jd, date.utcoffset)
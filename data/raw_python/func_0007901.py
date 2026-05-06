def fromJD(jd, utcoffset):
        """ Builds a Datetime object given a jd and utc offset. """
        if not isinstance(utcoffset, Time):
            utcoffset = Time(utcoffset)
        localJD = jd + utcoffset.value / 24.0
        date = Date(round(localJD))
        time = Time((localJD + 0.5 - date.jdn) * 24)
        return Datetime(date, time, utcoffset)
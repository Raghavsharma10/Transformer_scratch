def getyrdoy(date):
    """Return a tuple of year, day of year for a supplied datetime object."""

    try:
        doy = date.toordinal()-datetime(date.year,1,1).toordinal()+1
    except AttributeError:
        raise AttributeError("Must supply a pandas datetime object or " +
                             "equivalent")
    else:
        return date.year, doy
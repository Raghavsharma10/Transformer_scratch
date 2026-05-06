def premier_da_la_annee(jd):
    '''Determine the year in the French revolutionary calendar in which a given Julian day falls.
    Returns Julian day number containing fall equinox (first day of the FR year)'''
    p = ephem.previous_fall_equinox(dublin.from_jd(jd))
    previous = trunc(dublin.to_jd(p) - 0.5) + 0.5

    if previous + 364 < jd:
        # test if current day is the equinox if the previous equinox was a long time ago
        n = ephem.next_fall_equinox(dublin.from_jd(jd))
        nxt = trunc(dublin.to_jd(n) - 0.5) + 0.5

        if nxt <= jd:
            return nxt

    return previous
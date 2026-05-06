def _from_jd_equinox(jd):
    '''Calculate the FR day using the equinox as day 1'''
    jd = trunc(jd) + 0.5
    equinoxe = premier_da_la_annee(jd)

    an = gregorian.from_jd(equinoxe)[0] - YEAR_EPOCH
    mois = trunc((jd - equinoxe) / 30.) + 1
    jour = int((jd - equinoxe) % 30) + 1

    return (an, mois, jour)
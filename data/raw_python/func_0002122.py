def get_year_since_resonance(ringcube):
    "Calculate the fraction of the year since moon swap."
    t0 = dt(2006, 1, 21)
    td = ringcube.imagetime - t0
    return td.days / 365.25
def leq3(levels):
    """
    Calculates the energy-equivalent (Leq3) value
    given a regular measurement interval.
    """
    n = float(len(levels))
    sums = 0.0
    if sum(levels) == 0.0:
        return 0.0
    for l in levels:
        if l == 0:
            continue
        sums += pow(10.0, float(l) / 10.0)
    leq3 = 10.0 * math.log10((1.0 / n) * sums)
    leq3 = max(0.0, leq3)
    return leq3
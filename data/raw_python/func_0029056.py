def total_level(source_levels):
    """
    Calculates the total sound pressure level based on multiple source levels
    """
    sums = 0.0
    for l in source_levels:
        if l is None:
            continue
        if l == 0:
            continue
        sums += pow(10.0, float(l) / 10.0)
    level = 10.0 * math.log10(sums)
    return level
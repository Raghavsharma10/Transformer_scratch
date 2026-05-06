def pointbiserialr(cats, vals):
    """
Calculates a point-biserial correlation coefficient and the associated
probability value.  Taken from Heiman's Basic Statistics for the Behav.
Sci (1st), p.194.

Usage:   pointbiserialr(x,y)      where x,y are equal-length lists
Returns: Point-biserial r, two-tailed p-value
"""
    TINY = 1e-30
    if len(cats) != len(vals):
        raise ValueError('INPUT VALUES NOT PAIRED IN pointbiserialr.  ABORTING.')
    data = zip(cats, vals)
    categories = pstat.unique(cats)
    if len(categories) != 2:
        raise ValueError("Exactly 2 categories required for pointbiserialr().")
    else:   # there are 2 categories, continue
        c1 = [v for i, v in enumerate(vals) if cats[i] == categories[0]]
        c2 = [v for i, v in enumerate(vals) if cats[i] == categories[1]]
        xmean = mean(c1)
        ymean = mean(c2)
        n = len(vals)
        adjust = math.sqrt((len(c1) / float(n)) * (len(c2) / float(n)))
        rpb = (ymean - xmean) / samplestdev(vals) * adjust
        df = n - 2
        t = rpb * math.sqrt(df / ((1.0 - rpb + TINY) * (1.0 + rpb + TINY)))
        prob = betai(0.5 * df, 0.5, df / (df + t * t))  # t already a float
        return rpb, prob
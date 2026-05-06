def filter_for_ringspan(clearnacs, spanlimit):
    "filter for covered ringspan, giver in km."
    delta = clearnacs.MAXIMUM_RING_RADIUS - clearnacs.MINIMUM_RING_RADIUS
    f = delta < spanlimit
    ringspan = clearnacs[f].copy()
    return ringspan
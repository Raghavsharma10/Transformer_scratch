def estimateStellarTemperature(M_s):
    """Estimates stellar temperature using the main sequence relationship
    T ~ 5800*M^0.65 (Cox 2000).
    """
    # TODO improve with more x and k values from Cox 2000
    try:
        temp = (5800 * aq.K * float(M_s.rescale(aq.M_s) ** 0.65)).rescale(aq.K)
    except AttributeError:
        temp = np.nan
    return temp
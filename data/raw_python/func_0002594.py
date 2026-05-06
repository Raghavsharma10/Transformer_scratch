def aax(mt, x, m=1):
    """ äx : Returns the actuarial present value of an (immediate) annuity of 1 per time period 
    (whole life annuity-anticipatory). Payable 'm' per year at the beginning of the period 
    """
    return mt.Nx[x] / mt.Dx[x] - (float(m - 1) / float(m * 2))
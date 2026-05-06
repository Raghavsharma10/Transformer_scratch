def ax(mt, x, m=1):
    """ ax : Returns the actuarial present value of an (immediate) annuity of 1 per time period 
    (whole life annuity-late). Payable 'm' per year at the ends of the period 
    """
    return (mt.Nx[x] / mt.Dx[x] - 1) + (float(m - 1) / float(m * 2))
def taax(mt, x, t, m=1):
    """ n/äx : Return the actuarial present value of a deferred annuity (deferred n years): 
    n-year deferred whole life annuity-anticipatory. Payable 'm' per year at the beginning of the period 
    """
    return mt.Nx[x + t] / mt.Dx[x] - ((float(m - 1) / float(m * 2)) * (1 - nEx(mt, x, t)))
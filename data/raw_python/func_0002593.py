def aaxn(mt, x, n, m=1):
    """ äxn : Return the actuarial present value of a (immediate) temporal (term certain) annuity: 
    n-year temporary life annuity-anticipatory. Payable 'm' per year at the beginning of the period 
    """
    if m == 1:
        return (mt.Nx[x] - mt.Nx[x + n]) / mt.Dx[x]
    else:
        return (mt.Nx[x] - mt.Nx[x + n]) / mt.Dx[x] - ((float(m - 1) / float(m * 2)) * (1 - nEx(mt, x, n)))
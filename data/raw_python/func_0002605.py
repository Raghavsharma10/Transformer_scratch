def qaxn(mt, x, n, q, m=1):
    """ geometrica """
    q = float(q)
    j = (mt.i - q) / (1 + q)
    mtj = Actuarial(nt=mt.nt, i=j)
    return axn(mtj, x, n, m)
def qax(mt, x, q, m=1):
    """ geometrica """
    q = float(q)
    j = (mt.i - q) / (1 + q)
    mtj = Actuarial(nt=mt.nt, i=j)
    return ax(mtj, x, m)
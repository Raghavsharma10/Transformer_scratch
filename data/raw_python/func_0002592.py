def qAx(mt, x, q):
    """ This function evaluates the APV of a geometrically increasing annual annuity-due """
    q = float(q)
    j = (mt.i - q) / (1 + q)
    mtj = Actuarial(nt=mt.nt, i=j)
    return Ax(mtj, x)
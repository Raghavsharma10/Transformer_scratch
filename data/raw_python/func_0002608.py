def qtaax(mt, x, t, q, m=1):
    """ geometrica """
    q = float(q)
    j = (mt.i - q) / (1 + q)
    mtj = Actuarial(nt=mt.nt, i=j)
    return taax(mtj, x, t) - ((float(m - 1) / float(m * 2)) * (1 - nEx(mt, x, t)))
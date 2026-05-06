def qaaxn(mt, x, n, q, m = 1):
    """ geometrica """
    #i = float(nt[1])
    q = float(q)
    j = (mt.i - q) / (1 + q)
    mtj = Actuarial(nt=mt.nt, i=j)
    return aaxn(mtj, x, n, m)
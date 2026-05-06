def Mx(mt, x):
    """ Return the Mx """
    n = len(mt.Cx)
    sum1 = 0
    for j in range(x, n):
        k = mt.Cx[j]
        sum1 += k
    return sum1
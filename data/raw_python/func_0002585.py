def Sx(mt, x):
    """ Return the Sx """    
    n = len(mt.Nx)
    sum1 = 0
    for j in range(x, n):
        k = mt.Nx[j]
        sum1 += k
    return sum1
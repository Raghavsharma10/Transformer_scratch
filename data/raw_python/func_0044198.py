def getTauLambert(n):
    """Returns a list containing first n digits of Pi
    """
    myTau = tauGenLambert()
    result = []
    if n > 0:
        result += [next(myTau) for i in range(n)]
    myTau.close()
    return result
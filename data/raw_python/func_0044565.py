def getTauLeibniz(n):
    """Returns a list containing first n digits of Pi
    """
    myTau = tauGenLeibniz()
    result = []
    if n > 0:
        result += [next(myTau) for i in range(n)]
    myTau.close()
    return result
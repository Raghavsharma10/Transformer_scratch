def getPiLeibniz(n):
    """Returns a list containing first n digits of Pi
    """
    mypi = piGenLeibniz()
    result = []
    if n > 0:
        result += [next(mypi) for i in range(n)]
    mypi.close()
    return result
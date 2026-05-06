def piGenLeibniz():
    """A generator function that yields the digits of Pi
    """
    k = 1
    z = (1,0,0,1)
    while True:
        lft = __lfts(k)
        n = int(__next(z))
        if __safe(z,n):
            z = __prod(z,n)
            yield n
        else:
            z = __cons(z,lft)
            k += 1
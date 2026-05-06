def rabin_miller(p):
    """
    Performs a rabin-miller primality test

    :param p: Number to test
    :return: Bool of whether num is prime
    """
    # From this stackoverflow answer: https://codegolf.stackexchange.com/questions/26739/super-speedy-totient-function
    if p < 2:
        return False
    if p != 2 and p & 1 == 0:
        return False
    s = p - 1
    while s & 1 == 0:
        s >>= 1
    for x in range(10):
        a = random.randrange(p - 1) + 1
        temp = s
        mod = pow(a, temp, p)
        while temp != p - 1 and mod != 1 and mod != p - 1:
            mod = (mod * mod) % p
            temp = temp * 2
        if mod != p - 1 and temp % 2 == 0:
            return False
    return True
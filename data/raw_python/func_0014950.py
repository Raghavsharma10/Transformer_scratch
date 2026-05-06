def unfold(seed, by, last = __unique):
    """
    >>> list(unfold(1234, lambda x: divmod(x,10)))[::-1]
    [1, 2, 3, 4]
    >>> sum(imap(operator.mul,unfold(1234, lambda x:divmod(x,10)), iterate(lambda x:x*10)(1)))
    1234
    >>> g = unfold(1234, lambda x:divmod(x,10))
    >>> reduce((lambda (total,pow),digit:(total+pow*digit, 10*pow)), g, (0,1))
    (1234, 10000)
    """

    while True:
        seed, val = by(seed);
        if last == seed: return
        last = seed; yield val
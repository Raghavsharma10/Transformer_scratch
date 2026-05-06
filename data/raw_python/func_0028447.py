def drange(start: Decimal, stop: Decimal, num: int):
    '''
    A simplified version of numpy.linspace with default options
    '''
    delta = stop - start
    step = delta / (num - 1)
    yield from (start + step * Decimal(tick) for tick in range(0, num))
def range_expr(arg):
    '''
    Accepts a range expression which generates a range of values for a variable.

    Linear space range: "linspace:1,2,10" (start, stop, num) as in numpy.linspace
    Pythonic range: "range:1,10,2" (start, stop[, step]) as in Python's range
    Case range: "case:a,b,c" (comma-separated strings)
    '''
    key, value = arg.split('=', maxsplit=1)
    assert _rx_range_key.match(key), 'The key must be a valid slug string.'
    try:
        if value.startswith('case:'):
            return key, value[5:].split(',')
        elif value.startswith('linspace:'):
            start, stop, num = value[9:].split(',')
            return key, tuple(drange(Decimal(start), Decimal(stop), int(num)))
        elif value.startswith('range:'):
            range_args = map(int, value[6:].split(','))
            return key, tuple(range(*range_args))
        else:
            raise ArgumentTypeError('Unrecognized range expression type')
    except ValueError as e:
        raise ArgumentTypeError(str(e))
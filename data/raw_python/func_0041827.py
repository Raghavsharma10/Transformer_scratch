def get_units(unit, binary=False):
    ''' Sets the output unit and precision for future calculations and returns
        an integer and the string representation of it.
    '''
    result = None

    if unit == 'b':
        result = 1, 'Byte'

    elif binary:    # 2^X
        if   unit == 'k':
            result = 1024, 'Kibibyte'
        elif unit == 'm':
            result = 1048576, 'Mebibyte'
        elif unit == 'g':
            if opts.precision == -1:
                opts.precision = 3
            result = 1073741824, 'Gibibyte'
        elif unit == 't':
            if opts.precision == -1:
                opts.precision = 3
            result = 1099511627776, 'Tebibyte'

    else:           #  10^x
        if   unit == 'k':
            result = 1000, 'Kilobyte'
        elif unit == 'm':
            result = 1000000, 'Megabyte'
        elif unit == 'g':
            if opts.precision == -1:
                opts.precision = 3      # new defaults
            result = 1000000000, 'Gigabyte'
        elif unit == 't':
            if opts.precision == -1:
                opts.precision = 3
            result = 1000000000000, 'Terabyte'

    if not result:
        print(f'Warning: incorrect parameter: {unit}.')
        result = _outunit

    if opts.precision == -1:  # auto
        opts.precision = 0
    return result
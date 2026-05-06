def parse_atom(parse, minmax):
    """
    Returns a set containing valid values for a given cron-style range of
    numbers. The 'minmax' arguments is a two element iterable containing the
    inclusive upper and lower limits of the expression.

    Examples:
    >>> parse_atom("1-5",(0,6))
    set([1, 2, 3, 4, 5])

    >>> parse_atom("*/6",(0,23))
    set([0, 6, 12, 18])

    >>> parse_atom("18-6/4",(0,23))
    set([18, 22, 0, 4])

    >>> parse_atom("*/9",(0,23))
    set([0, 9, 18])
    """
    parse = parse.strip()
    increment = 1
    if parse == '*':
        return set(xrange(minmax[0], minmax[1] + 1))
    elif parse.isdigit():
        # A single number still needs to be returned as a set
        value = int(parse)
        if value >= minmax[0] and value <= minmax[1]:
            return set((value,))
        else:
            raise ValueError("\"%s\" is not within valid range." % parse)
    elif '-' in parse or '/' in parse:
        divide = parse.split('/')
        subrange = divide[0]
        if len(divide) == 2:
            # Example: 1-3/5 or */7 increment should be 5 and 7 respectively
            increment = int(divide[1])

        if '-' in subrange:
            # Example: a-b
            prefix, suffix = [int(n) for n in subrange.split('-')]
            if prefix < minmax[0] or suffix > minmax[1]:
                raise ValueError("\"%s\" is not within valid range." % parse)
        elif subrange.isdigit():
            # Handle offset increments e.g. 5/15 to run at :05, :20, :35, and :50
            return set(xrange(int(subrange), minmax[1] + 1, increment))
        elif subrange == '*':
            # Include all values with the given range
            prefix, suffix = minmax
        else:
            raise ValueError("Unrecognized symbol \"%s\"" % subrange)

        if prefix < suffix:
            # Example: 7-10
            return set(xrange(prefix, suffix + 1, increment))
        else:
            # Example: 12-4/2; (12, 12 + n, ..., 12 + m*n) U (n_0, ..., 4)
            noskips = list(xrange(prefix, minmax[1] + 1))
            noskips += list(xrange(minmax[0], suffix + 1))
            return set(noskips[::increment])
    else:
        raise ValueError("Atom \"%s\" not in a recognized format." % parse)
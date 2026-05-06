def best_tile_layout(pool_size):
    """ Determine and return the best layout of "tiles" for fastest
    overall parallel processing of a rectangular image broken up into N
    smaller equally-sized rectangular tiles, given as input the number
    of processes/chunks which can be run/worked at the same time (pool_size).

    This attempts to return a layout whose total number of tiles is as
    close as possible to pool_size, without going over (and thus not
    really taking advantage of pooling).  Since we can vary the
    size of the rectangles, there is not much (any?) benefit to pooling.

    Returns a tuple of ( <num tiles in X dir>, <num in Y direction> )

    This assumes the image in question is relatively close to square, and
    so the returned tuple attempts to give a layout which is as
    squarishly-blocked as possible, except in cases where speed would be
    sacrificed.

    EXAMPLES:

    For pool_size of 4, the best result is 2x2.

    For pool_size of 6, the best result is 2x3.

    For pool_size of 5, a result of 1x5 is better than a result of
    2x2 (which would leave one core unused), and 1x5 is also better than
    a result of 2x3 (which would require one core to work twice while all
    others wait).

    For higher, odd pool_size values (say 39), it is deemed best to
    sacrifice a few unused cores to satisfy our other constraints, and thus
    the result of 6x6 is best (giving 36 tiles and 3 unused cores).
    """
    # Easy answer sanity-checks
    if pool_size < 2:
        return (1, 1)

    # Next, use a small mapping of hard-coded results.  While we agree
    # that many of these are unlikely pool_size values, they are easy
    # to accomodate.
    mapping = { 0:(1,1), 1:(1,1), 2:(1,2), 3:(1,3), 4:(2,2), 5:(1,5),
                6:(2,3), 7:(2,3), 8:(2,4), 9:(3,3), 10:(2,5), 11:(2,5),
                14:(2,7), 18:(3,6), 19:(3,6), 28:(4,7), 29:(4,7),
                32:(4,8), 33:(4,8), 34:(4,8), 40:(4,10), 41:(4,10) }
    if pool_size in mapping:
        return mapping[pool_size]

    # Next, take a guess using the square root and (for the sake of
    # simplicity), go with it.  We *could* get much fancier here...
    # Use floor-rounding (not ceil) so that the total number of resulting
    # tiles is <= pool_size.
    xnum = int(math.sqrt(pool_size))
    ynum = int((1.*pool_size)/xnum)
    return (xnum, ynum)
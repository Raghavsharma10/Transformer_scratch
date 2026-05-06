def _totuple( x ):
    """Utility stuff to convert string, int, long, float, None or anything to a usable tuple."""

    if isinstance( x, basestring ):
        out = x,
    elif isinstance( x, ( int, long, float ) ):
        out = str( x ),
    elif x is None:
        out = None,
    else:
        out = tuple( x )

    return out
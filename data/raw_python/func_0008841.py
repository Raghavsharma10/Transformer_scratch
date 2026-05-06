def fix_shape(source):
    """
    Ensure that a>=b for a given source object.
    If a<b then swap a/b and increment pa by 90.
    err_a/err_b are also swapped as needed.

    Parameters
    ----------
    source : object
        any object with a/b/pa/err_a/err_b properties

    """
    if source.a < source.b:
        source.a, source.b = source.b, source.a
        source.err_a, source.err_b = source.err_b, source.err_a
        source.pa += 90
    return
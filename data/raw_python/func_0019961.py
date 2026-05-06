def StudentT(v, tag=None):
    """
    A Student-T random variate
    
    Parameters
    ----------
    v : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert int(v) == v and v >= 1, 'Student-T "v" must be an integer greater than 0'
    return uv(ss.t(v), tag=tag)
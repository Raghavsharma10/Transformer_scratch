def antenna_uvw(uvw, antenna1, antenna2, chunks,
                nr_of_antenna, check_missing=False,
                check_decomposition=False, max_err=100):
    """
    Computes per-antenna UVW coordinates from baseline ``uvw``,
    ``antenna1`` and ``antenna2`` coordinates logically grouped
    into baseline chunks.

    The example below illustrates two baseline chunks
    of size 6 and 5, respectively.

    .. code-block:: python

        uvw = ...
        ant1 = np.array([0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1], dtype=np.int32)
        ant2 = np.array([1, 2, 3, 2, 3, 3, 1, 2, 3, 1, 2], dtype=np.int32)
        chunks = np.array([6, 5], dtype=np.int32)

        ant_uv = antenna_uvw(uvw, ant1, ant2, chunks, nr_of_antenna=4)

    The first antenna of the first baseline of a chunk is chosen as the origin
    of the antenna coordinate system, while the second antenna is set to the
    negative of the baseline UVW coordinate. Subsequent antenna UVW coordinates
    are iteratively derived from the first two coordinates. Thus,
    the baseline indices need not be properly ordered (within the chunk).

    If it is not possible to derive coordinates for an antenna,
    it's coordinate will be set to nan.

    Parameters
    ----------
    uvw : np.ndarray
        Baseline UVW coordinates of shape (row, 3)
    antenna1 : np.ndarray
        Baseline first antenna of shape (row,)
    antenna2 : np.ndarray
        Baseline second antenna of shape (row,)
    chunks : np.ndarray
        Number of baselines per unique timestep with shape (chunks,)
        :code:`np.sum(chunks) == row` should hold.
    nr_of_antenna : int
        Total number of antenna in the solution.
    check_missing (optional) : bool
        If ``True`` raises an exception if it was not possible
        to compute UVW coordinates for all antenna (i.e. some were nan).
        Defaults to ``False``.
    check_decomposition (optional) : bool
        If ``True``, checks that the antenna decomposition accurately
        reproduces the coordinates in ``uvw``, or that
        :code:`ant_uvw[c,ant1,:] - ant_uvw[c,ant2,:] == uvw[s:e,:]`
        where ``s`` and ``e`` are the start and end rows
        of chunk ``c`` respectively. Defaults to ``False``.
    max_err (optional) : integer
        Maximum numbers of errors when checking for missing antenna
        or innacurate decompositions. Defaults to ``100``.

    Returns
    -------
    np.ndarray
        Antenna UVW coordinates of shape (chunks, nr_of_antenna, 3)
    """

    ant_uvw = _antenna_uvw(uvw, antenna1, antenna2, chunks, nr_of_antenna)

    if check_missing:
        _raise_missing_antenna_errors(ant_uvw, max_err=max_err)

    if check_decomposition:
        _raise_decomposition_errors(uvw, antenna1, antenna2, chunks,
                                    ant_uvw, max_err=max_err)

    return ant_uvw
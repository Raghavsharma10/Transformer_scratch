def _antenna_uvw(uvw, antenna1, antenna2, chunks, nr_of_antenna):
    """ numba implementation of antenna_uvw """

    if antenna1.ndim != 1:
        raise ValueError("antenna1 shape should be (row,)")

    if antenna2.ndim != 1:
        raise ValueError("antenna2 shape should be (row,)")

    if uvw.ndim != 2 or uvw.shape[1] != 3:
        raise ValueError("uvw shape should be (row, 3)")

    if not (uvw.shape[0] == antenna1.shape[0] == antenna2.shape[0]):
        raise ValueError("First dimension of uvw, antenna1 "
                         "and antenna2 do not match")

    if chunks.ndim != 1:
        raise ValueError("chunks shape should be (utime,)")

    if nr_of_antenna < 1:
        raise ValueError("nr_of_antenna < 1")

    ant_uvw_shape = (chunks.shape[0], nr_of_antenna, 3)
    antenna_uvw = np.full(ant_uvw_shape, np.nan, dtype=uvw.dtype)

    start = 0

    for ci, chunk in enumerate(chunks):
        end = start + chunk

        # one pass should be enough!
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw, ci, start, end)

        start = end

    return antenna_uvw
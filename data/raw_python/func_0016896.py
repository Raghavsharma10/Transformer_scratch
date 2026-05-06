def _raise_decomposition_errors(uvw, antenna1, antenna2,
                                chunks, ant_uvw, max_err):
    """ Raises informative exception for an invalid decomposition """

    start = 0

    problem_str = []

    for ci, chunk in enumerate(chunks):
        end = start + chunk

        ant1 = antenna1[start:end]
        ant2 = antenna2[start:end]
        cuvw = uvw[start:end]

        ant1_uvw = ant_uvw[ci, ant1, :]
        ant2_uvw = ant_uvw[ci, ant2, :]
        ruvw = ant2_uvw - ant1_uvw

        # Identifty rows where any of the UVW components differed
        close = np.isclose(ruvw, cuvw)
        problems = np.nonzero(np.logical_or.reduce(np.invert(close), axis=1))

        for row in problems[0]:
            problem_str.append("[row %d [%d, %d] (chunk %d)]: "
                               "original %s recovered %s "
                               "ant1 %s ant2 %s" % (
                                    start+row, ant1[row], ant2[row], ci,
                                    cuvw[row], ruvw[row],
                                    ant1_uvw[row], ant2_uvw[row]))

            # Exit inner loop early
            if len(problem_str) >= max_err:
                break

        # Exit outer loop early
        if len(problem_str) >= max_err:
            break

        start = end

    # Return early if nothing was wrong
    if len(problem_str) == 0:
        return

    # Add a preamble and raise exception
    problem_str = ["Antenna UVW Decomposition Failed",
                   "The following differences were found "
                   "(first 100):"] + problem_str
    raise AntennaUVWDecompositionError('\n'.join(problem_str))
def _raise_missing_antenna_errors(ant_uvw, max_err):
    """ Raises an informative error for missing antenna """

    # Find antenna uvw coordinates where any UVW component was nan
    # nan + real == nan
    problems = np.nonzero(np.add.reduce(np.isnan(ant_uvw), axis=2))
    problem_str = []

    for c, a in zip(*problems):
        problem_str.append("[chunk %d antenna %d]" % (c, a))

        # Exit early
        if len(problem_str) >= max_err:
            break

    # Return early if nothing was wrong
    if len(problem_str) == 0:
        return

    # Add a preamble and raise exception
    problem_str = ["Antenna were missing"] + problem_str
    raise AntennaMissingError('\n'.join(problem_str))
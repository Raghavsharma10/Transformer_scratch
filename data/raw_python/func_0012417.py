def check_ns_run_logls(run, dup_assert=False, dup_warn=False):
    """Check run logls are unique and in the correct order.

    Parameters
    ----------
    run: dict
        nested sampling run to check.
    dup_assert: bool, optional
        Whether to raise and AssertionError if there are duplicate logl values.
    dup_warn: bool, optional
        Whether to give a UserWarning if there are duplicate logl values (only
        used if dup_assert is False).

    Raises
    ------
    AssertionError
        if run does not have expected properties.
    """
    assert np.array_equal(run['logl'], run['logl'][np.argsort(run['logl'])])
    if dup_assert or dup_warn:
        unique_logls, counts = np.unique(run['logl'], return_counts=True)
        repeat_logls = run['logl'].shape[0] - unique_logls.shape[0]
        msg = ('{} duplicate logl values (out of a total of {}). This may be '
               'caused by limited numerical precision in the output files.'
               '\nrepeated logls = {}\ncounts = {}\npositions in list of {}'
               ' unique logls = {}').format(
                   repeat_logls, run['logl'].shape[0],
                   unique_logls[counts != 1], counts[counts != 1],
                   unique_logls.shape[0], np.where(counts != 1)[0])
        if dup_assert:
            assert repeat_logls == 0, msg
        elif dup_warn:
            if repeat_logls != 0:
                warnings.warn(msg, UserWarning)
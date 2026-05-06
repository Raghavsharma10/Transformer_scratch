def check_ns_run(run, dup_assert=False, dup_warn=False):
    """Checks a nestcheck format nested sampling run dictionary has the
    expected properties (see the data_processing module docstring for more
    details).

    Parameters
    ----------
    run: dict
        nested sampling run to check.
    dup_assert: bool, optional
        See check_ns_run_logls docstring.
    dup_warn: bool, optional
        See check_ns_run_logls docstring.


    Raises
    ------
    AssertionError
        if run does not have expected properties.
    """
    assert isinstance(run, dict)
    check_ns_run_members(run)
    check_ns_run_logls(run, dup_assert=dup_assert, dup_warn=dup_warn)
    check_ns_run_threads(run)
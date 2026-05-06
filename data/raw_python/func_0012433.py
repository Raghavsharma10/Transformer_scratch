def estimator_values_df(run_list, estimator_list, **kwargs):
    """Get a dataframe of estimator values.

    NB when parallelised the results will not be produced in order (so results
    from some run number will not nessesarily correspond to that number run in
    run_list).

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs, optional
        Name of each func in estimator_list.
    parallel: bool, optional
        Whether or not to parallelise - see parallel_utils.parallel_apply.
    save_name: str or None, optional
        See nestcheck.io_utils.save_load_result.
    save: bool, optional
        See nestcheck.io_utils.save_load_result.
    load: bool, optional
        See nestcheck.io_utils.save_load_result.
    overwrite_existing: bool, optional
        See nestcheck.io_utils.save_load_result.

    Returns
    -------
    df: pandas DataFrame
        Results table showing calculation values and diagnostics. Rows
        show different runs.
        Columns have titles given by estimator_names and show results for the
        different functions in estimators_list.
    """
    estimator_names = kwargs.pop(
        'estimator_names',
        ['est_' + str(i) for i in range(len(estimator_list))])
    parallel = kwargs.pop('parallel', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    values_list = pu.parallel_apply(
        nestcheck.ns_run_utils.run_estimators, run_list,
        func_args=(estimator_list,), parallel=parallel)
    df = pd.DataFrame(np.stack(values_list, axis=0))
    df.columns = estimator_names
    df.index.name = 'run'
    return df
def thread_values_df(run_list, estimator_list, estimator_names, **kwargs):
    """Calculates estimator values for the constituent threads of the input
    runs.

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs
        Name of each func in estimator_list.
    kwargs:
        Kwargs to pass to parallel_apply.

    Returns
    -------
    df: pandas data frame
        Columns represent estimators and rows represent runs.
        Each cell contains a 1d numpy array with length equal to the number
        of threads in the run, containing the results from evaluating the
        estimator on each thread.
    """
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'desc': 'thread values'})
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    # get thread results
    thread_vals_arrays = pu.parallel_apply(
        nestcheck.error_analysis.run_thread_values, run_list,
        func_args=(estimator_list,), tqdm_kwargs=tqdm_kwargs, **kwargs)
    df = pd.DataFrame()
    for i, name in enumerate(estimator_names):
        df[name] = [arr[i, :] for arr in thread_vals_arrays]
    # Check there are the correct number of thread values in each cell
    for vals_shape in df.loc[0].apply(lambda x: x.shape).values:
        assert vals_shape == (run_list[0]['thread_min_max'].shape[0],), \
            ('Should be nlive=' + str(run_list[0]['thread_min_max'].shape[0]) +
             ' values in each cell. The cell contains array with shape ' +
             str(vals_shape))
    return df
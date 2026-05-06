def bs_values_df(run_list, estimator_list, estimator_names, n_simulate,
                 **kwargs):
    """Computes a data frame of bootstrap resampled values.

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs
        Name of each func in estimator_list.
    n_simulate: int
        Number of bootstrap replications to use on each run.
    kwargs:
        Kwargs to pass to parallel_apply.

    Returns
    -------
    bs_values_df: pandas data frame
        Columns represent estimators and rows represent runs.
        Each cell contains a 1d array of bootstrap resampled values for the run
        and estimator.
    """
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'desc': 'bs values'})
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    bs_values_list = pu.parallel_apply(
        nestcheck.error_analysis.run_bootstrap_values, run_list,
        func_args=(estimator_list,), func_kwargs={'n_simulate': n_simulate},
        tqdm_kwargs=tqdm_kwargs, **kwargs)
    df = pd.DataFrame()
    for i, name in enumerate(estimator_names):
        df[name] = [arr[i, :] for arr in bs_values_list]
    # Check there are the correct number of bootstrap replications in each cell
    for vals_shape in df.loc[0].apply(lambda x: x.shape).values:
        assert vals_shape == (n_simulate,), (
            'Should be n_simulate=' + str(n_simulate) + ' values in ' +
            'each cell. The cell contains array with shape ' +
            str(vals_shape))
    return df
def run_list_error_values(run_list, estimator_list, estimator_names,
                          n_simulate=100, **kwargs):
    """Gets a data frame with calculation values and error diagnostics for each
    run in the input run list.

    NB when parallelised the results will not be produced in order (so results
    from some run number will not nessesarily correspond to that number run in
    run_list).

    Parameters
    ----------
    run_list: list of dicts
        List of nested sampling run dicts.
    estimator_list: list of functions
        Estimators to apply to runs.
    estimator_names: list of strs
        Name of each func in estimator_list.
    n_simulate: int, optional
        Number of bootstrap replications to use on each run.
    thread_pvalue: bool, optional
        Whether or not to compute KS test diaganostic for correlations between
        threads within a run.
    bs_stat_dist: bool, optional
        Whether or not to compute statistical distance between bootstrap error
        distributions diaganostic.
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
        show different runs (or pairs of runs for pairwise comparisons).
        Columns have titles given by estimator_names and show results for the
        different functions in estimators_list.
    """
    thread_pvalue = kwargs.pop('thread_pvalue', False)
    bs_stat_dist = kwargs.pop('bs_stat_dist', False)
    parallel = kwargs.pop('parallel', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert len(estimator_list) == len(estimator_names), (
        'len(estimator_list) = {0} != len(estimator_names = {1}'
        .format(len(estimator_list), len(estimator_names)))
    # Calculation results
    # -------------------
    df = estimator_values_df(run_list, estimator_list, parallel=parallel,
                             estimator_names=estimator_names)
    df.index = df.index.map(str)
    df['calculation type'] = 'values'
    df.set_index('calculation type', drop=True, append=True, inplace=True)
    df = df.reorder_levels(['calculation type', 'run'])
    # Bootstrap stds
    # --------------
    # Create bs_vals_df then convert to stds so bs_vals_df does not need to be
    # recomputed if bs_stat_dist is True
    bs_vals_df = bs_values_df(run_list, estimator_list, estimator_names,
                              n_simulate, parallel=parallel)
    bs_std_df = bs_vals_df.applymap(lambda x: np.std(x, ddof=1))
    bs_std_df.index.name = 'run'
    bs_std_df['calculation type'] = 'bootstrap std'
    bs_std_df.set_index('calculation type', drop=True, append=True,
                        inplace=True)
    bs_std_df = bs_std_df.reorder_levels(['calculation type', 'run'])
    df = pd.concat([df, bs_std_df])
    # Pairwise KS p-values on threads
    # -------------------------------
    if thread_pvalue:
        t_vals_df = thread_values_df(
            run_list, estimator_list, estimator_names, parallel=parallel)
        t_d_df = pairwise_dists_on_cols(t_vals_df, earth_mover_dist=False,
                                        energy_dist=False)
        # Keep only the p value not the distance measures
        t_d_df = t_d_df.xs('ks pvalue', level='calculation type',
                           drop_level=False)
        # Append 'thread ' to caclulcation type
        t_d_df.index.set_levels(['thread ks pvalue'], level='calculation type',
                                inplace=True)
        df = pd.concat([df, t_d_df])
    # Pairwise distances on BS distributions
    # --------------------------------------
    if bs_stat_dist:
        b_d_df = pairwise_dists_on_cols(bs_vals_df)
        # Select only statistical distances - not KS pvalue as this is not
        # useful for the bootstrap resample distributions (see Higson et al.
        # 2019 for more details).
        dists = ['ks distance', 'earth mover distance', 'energy distance']
        b_d_df = b_d_df.loc[pd.IndexSlice[dists, :], :]
        # Append 'bootstrap ' to caclulcation type
        new_ind = ['bootstrap ' +
                   b_d_df.index.get_level_values('calculation type'),
                   b_d_df.index.get_level_values('run')]
        b_d_df.set_index(new_ind, inplace=True)
        df = pd.concat([df, b_d_df])
    return df
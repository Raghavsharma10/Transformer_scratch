def efficiency_gain_df(method_names, method_values, est_names, **kwargs):
    r"""Calculated data frame showing

    .. math::

        \mathrm{efficiency\,gain}
        =
        \frac{\mathrm{Var[base\,method]}}{\mathrm{Var[new\,method]}}

    See the dynamic nested sampling paper (Higson et al. 2019) for more
    details.

    The standard method on which to base the gain is assumed to be the first
    method input.

    The output DataFrame will contain rows:
        mean [dynamic goal]: mean calculation result for standard nested
            sampling and dynamic nested sampling with each input dynamic
            goal.
        std [dynamic goal]: standard deviation of results for standard
            nested sampling and dynamic nested sampling with each input
            dynamic goal.
        gain [dynamic goal]: the efficiency gain (computational speedup)
            from dynamic nested sampling compared to standard nested
            sampling. This equals (variance of standard results) /
            (variance of dynamic results); see the dynamic nested
            sampling paper for more details.

    Parameters
    ----------
    method names: list of strs
    method values: list
        Each element is a list of 1d arrays of results for the method. Each
        array must have shape (len(est_names),).
    est_names: list of strs
        Provide column titles for output df.
    true_values: iterable of same length as estimators list
        True values of the estimators for the given likelihood and prior.

    Returns
    -------
    results: pandas data frame
        Results data frame.
    """
    true_values = kwargs.pop('true_values', None)
    include_true_values = kwargs.pop('include_true_values', False)
    include_rmse = kwargs.pop('include_rmse', False)
    adjust_nsamp = kwargs.pop('adjust_nsamp', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if adjust_nsamp is not None:
        assert adjust_nsamp.shape == (len(method_names),)
    assert len(method_names) == len(method_values)
    df_dict = {}
    for i, method_name in enumerate(method_names):
        # Set include_true_values=False as we don't want them repeated for
        # every method
        df = summary_df_from_list(
            method_values[i], est_names, true_values=true_values,
            include_true_values=False, include_rmse=include_rmse)
        if i != 0:
            stats = ['std']
            if include_rmse:
                stats.append('rmse')
            if adjust_nsamp is not None:
                # Efficiency gain measures performance per number of
                # samples (proportional to computational work). If the
                # number of samples is not the same we can adjust this.
                adjust = (adjust_nsamp[0] / adjust_nsamp[i])
            else:
                adjust = 1
            for stat in stats:
                # Calculate efficiency gain vs standard nested sampling
                gain, gain_unc = get_eff_gain(
                    df_dict[method_names[0]].loc[(stat, 'value')],
                    df_dict[method_names[0]].loc[(stat, 'uncertainty')],
                    df.loc[(stat, 'value')],
                    df.loc[(stat, 'uncertainty')], adjust=adjust)
                key = stat + ' efficiency gain'
                df.loc[(key, 'value'), :] = gain
                df.loc[(key, 'uncertainty'), :] = gain_unc
        df_dict[method_name] = df
    results = pd.concat(df_dict)
    results.index.rename('dynamic settings', level=0, inplace=True)
    new_ind = []
    new_ind.append(pd.CategoricalIndex(
        results.index.get_level_values('calculation type'), ordered=True,
        categories=['true values', 'mean', 'std', 'rmse',
                    'std efficiency gain', 'rmse efficiency gain']))
    new_ind.append(pd.CategoricalIndex(
        results.index.get_level_values('dynamic settings'),
        ordered=True, categories=[''] + method_names))
    new_ind.append(results.index.get_level_values('result type'))
    results.set_index(new_ind, inplace=True)
    if include_true_values:
        with warnings.catch_warnings():
            # Performance not an issue here so suppress annoying warning
            warnings.filterwarnings('ignore', message=(
                'indexing past lexsort depth may impact performance.'))
            results.loc[('true values', '', 'value'), :] = true_values
    results.sort_index(inplace=True)
    return results
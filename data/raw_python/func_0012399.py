def summary_df(df_in, **kwargs):
    """Make a panda data frame of the mean and std devs of an array of results,
    including the uncertainties on the values.

    This is similar to pandas.DataFrame.describe but also includes estimates of
    the numerical uncertainties.

    The output DataFrame has multiindex levels:

    'calculation type':  mean and standard deviations of the data.
    'result type': value and uncertainty for each quantity.

    calculation type    result type        column_1     column_2      ...
    mean                value
    mean                uncertainty
    std                 value
    std                 uncertainty

    Parameters
    ----------
    df_in: pandas DataFrame
    true_values: array
        Analytical values if known for comparison with mean. Used to
        calculate root mean squared errors (RMSE).
    include_true_values: bool, optional
        Whether or not to include true values in the output DataFrame.
    include_rmse: bool, optional
        Whether or not to include root-mean-squared-errors in the output
        DataFrame.


    Returns
    -------
    df: MultiIndex DataFrame
    """
    true_values = kwargs.pop('true_values', None)
    include_true_values = kwargs.pop('include_true_values', False)
    include_rmse = kwargs.pop('include_rmse', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if true_values is not None:
        assert true_values.shape[0] == df_in.shape[1], (
            'There should be one true value for every column! '
            'true_values.shape=' + str(true_values.shape) + ', '
            'df_in.shape=' + str(df_in.shape))
    # make the data frame
    df = pd.DataFrame([df_in.mean(axis=0), df_in.std(axis=0, ddof=1)],
                      index=['mean', 'std'])
    if include_true_values:
        assert true_values is not None
        df.loc['true values'] = true_values
    # Make index categorical to allow sorting
    df.index = pd.CategoricalIndex(df.index.values, ordered=True,
                                   categories=['true values', 'mean', 'std',
                                               'rmse'],
                                   name='calculation type')
    # add uncertainties
    num_cals = df_in.shape[0]
    mean_unc = df.loc['std'] / np.sqrt(num_cals)
    std_unc = df.loc['std'] * np.sqrt(1 / (2 * (num_cals - 1)))
    df['result type'] = pd.Categorical(['value'] * df.shape[0], ordered=True,
                                       categories=['value', 'uncertainty'])
    df.set_index(['result type'], drop=True, append=True, inplace=True)
    df.loc[('mean', 'uncertainty'), :] = mean_unc.values
    df.loc[('std', 'uncertainty'), :] = std_unc.values
    if include_rmse:
        assert true_values is not None, \
            'Need to input true values for RMSE!'
        rmse, rmse_unc = rmse_and_unc(df_in.values, true_values)
        df.loc[('rmse', 'value'), :] = rmse
        df.loc[('rmse', 'uncertainty'), :] = rmse_unc
    # Ensure correct row order by sorting
    df.sort_index(inplace=True)
    # Cast calculation type index back from categorical to string to allow
    # adding new calculation types
    df.set_index(
        [df.index.get_level_values('calculation type').astype(str),
         df.index.get_level_values('result type')],
        inplace=True)
    return df
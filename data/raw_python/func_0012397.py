def summary_df_from_list(results_list, names, **kwargs):
    """Make a panda data frame of the mean and std devs of each element of a
    list of 1d arrays, including the uncertainties on the values.

    This just converts the array to a DataFrame and calls summary_df on it.

    Parameters
    ----------
    results_list: list of 1d numpy arrays
        Must have same length as names.
    names: list of strs
        Names for the output df's columns.
    kwargs: dict, optional
        Keyword arguments to pass to summary_df.

    Returns
    -------
    df: MultiIndex DataFrame
        See summary_df docstring for more details.
    """
    for arr in results_list:
        assert arr.shape == (len(names),)
    df = pd.DataFrame(np.stack(results_list, axis=0))
    df.columns = names
    return summary_df(df, **kwargs)
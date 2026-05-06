def summary_df_from_array(results_array, names, axis=0, **kwargs):
    """Make a panda data frame of the mean and std devs of an array of results,
    including the uncertainties on the values.

    This function converts the array to a DataFrame and calls summary_df on it.

    Parameters
    ----------
    results_array: 2d numpy array
    names: list of str
        Names for the output df's columns.
    axis: int, optional
        Axis on which to calculate summary statistics.

    Returns
    -------
    df: MultiIndex DataFrame
        See summary_df docstring for more details.
    """
    assert axis == 0 or axis == 1
    df = pd.DataFrame(results_array)
    if axis == 1:
        df = df.T
    df.columns = names
    return summary_df(df, **kwargs)
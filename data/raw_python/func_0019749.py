def df2list(df):
    """
    Convert a MultiIndex df to list

    Parameters
    ----------

    df : pandas.DataFrame
        A MultiIndex DataFrame where the first level is subjects and the second
        level is lists (e.g. egg.pres)

    Returns
    ----------

    lst : a list of lists of lists of values
        The input df reformatted as a list

    """
    subjects = df.index.levels[0].values.tolist()
    lists = df.index.levels[1].values.tolist()
    idx = pd.IndexSlice
    df = df.loc[idx[subjects,lists],df.columns]
    lst = [df.loc[sub,:].values.tolist() for sub in subjects]
    return lst
def none_missing(df, columns=None):
    """
    Asserts that there are no missing values (NaNs) in the DataFrame.
    """
    if columns is None:
        columns = df.columns
    assert not df[columns].isnull().any().any()
    return df
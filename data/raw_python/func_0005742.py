def make_exploded_column(df, colname_new, colname_old):
    """
    Internal helper function used by `explode_columns()`.
    """
    s = df[colname_old].apply(pd.Series).stack()
    s.name = colname_new
    return s
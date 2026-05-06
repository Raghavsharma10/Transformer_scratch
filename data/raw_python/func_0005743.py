def explode_columns(df, colnames):
    """
    Given a dataframe with certain columns that contain lists,
    return another dataframe where the elements in each list
    are "exploded" into individual rows.

    Example:

    >>> df
       col1 col2             col3          col4
    0   foo   11  [DDD, AAA, CCC]  [dd, aa, cc]
    1   bar   22            [FFF]          [ff]
    2  quux   33       [EEE, BBB]      [ee, bb]

    >>> explode_columns(df, ['col3'])
       col1 col2 col3 col4
    0   foo   11  DDD   dd
    1   foo   11  AAA   aa
    2   foo   11  CCC   cc
    3   bar   22  FFF   ff
    4  quux   33  EEE   ee
    5  quux   33  BBB   bb

    >>> explode_columns(df, {'col3_exploded': 'col3'})
       col1 col2 col3_exploded col4
    0   foo   11           DDD   dd
    1   foo   11           AAA   aa
    2   foo   11           CCC   cc
    3   bar   22           FFF   ff
    4  quux   33           EEE   ee
    5  quux   33           BBB   bb
    """
    if isinstance(colnames, (list, tuple)):
        colnames = {name: name for name in colnames}

    remaining_columns = list(df.columns.difference(colnames.values()))
    df2 = df.set_index(remaining_columns)
    df3 = pd.concat((make_exploded_column(df2, col_new, col_old) for col_new, col_old in colnames.items()), axis=1)
    levels_to_reset = list(range(len(remaining_columns)))
    df4 = df3.reset_index(level=levels_to_reset).reset_index(drop=True)
    return df4
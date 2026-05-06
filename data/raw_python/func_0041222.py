def table_nan_locs(table):
    """
    from http://stackoverflow.com/a/14033137/623735
    # gets the indices of the rows with nan values in a dataframe
    pd.isnull(df).any(1).nonzero()[0]
    """
    ans = []
    for rownum, row in enumerate(table):
        try:
            if pd.isnull(row).any():
                colnums = pd.isnull(row).nonzero()[0]
                ans += [(rownum, colnum) for colnum in colnums]
        except AttributeError:  # table is really just a sequence of scalars
            if pd.isnull(row):
                ans += [(rownum, 0)]
    return ans
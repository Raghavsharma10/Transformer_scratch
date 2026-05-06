def dataset_nan_locs(ds):
    """
    from http://stackoverflow.com/a/14033137/623735
    # gets the indices of the rows with nan values in a dataframe
    pd.isnull(df).any(1).nonzero()[0]
    """
    ans = []
    for sampnum, sample in enumerate(ds):
        if pd.isnull(sample).any():
            ans += [{
                'sample': sampnum,
                'input':  pd.isnull(sample[0]).nonzero()[0],
                'output': pd.isnull(sample[1]).nonzero()[0],
                }]
    return ans
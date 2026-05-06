def combine_dfs(dfs, names, method):
    """Combine dataframes.

    Combination is either done simple by just concatenating the DataFrames
    or performs tracking by adding the name of the dataset as a column."""
    if method == "track":
        res = list()
        for df, identifier in zip(dfs, names):
            df["dataset"] = identifier
            res.append(df)
        return pd.concat(res, ignore_index=True)
    elif method == "simple":
        return pd.concat(dfs, ignore_index=True)
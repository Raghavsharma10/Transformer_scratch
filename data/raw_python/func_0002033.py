def print_db_stats():
    """Print database stats.

    Returns
    -------
    pd.DataFrame
        Table with the found data items per type.
    """
    dbroot = get_db_root()
    n_ids = len(list(dbroot.glob("[N,W]*")))
    print("Number of WACs and NACs in database: {}".format(n_ids))
    print("These kind of data are in the database: (returning pd.DataFrame)")
    d = {}
    for key, val in PathManager.extensions.items():
        d[key] = [len(list(dbroot.glob("**/*" + val)))]
    return pd.DataFrame(d)
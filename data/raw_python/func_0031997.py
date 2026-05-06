def save_data(data, file_fmt, append=False, drop_dups=None, info=None, **kwargs):
    """
    Save data to file

    Args:
        data: pd.DataFrame
        file_fmt: data file format in terms of f-strings
        append: if append data to existing data
        drop_dups: list, drop duplicates in columns
        info: dict, infomation to be hashed and passed to f-strings
        **kwargs: additional parameters for f-strings

    Examples:
        >>> data = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
        >>> # save_data(
        >>>     # data, '{ROOT}/daily/{typ}.parq',
        >>>     # ROOT='tests/data', typ='earnings'
        >>> # )
    """
    d_file = data_file(file_fmt=file_fmt, info=info, **kwargs)
    if append and files.exists(d_file):
        data = pd.DataFrame(pd.concat([pd.read_parquet(d_file), data], sort=False))
        if drop_dups is not None:
            data.drop_duplicates(subset=utils.tolist(drop_dups), inplace=True)

    if not data.empty: data.to_parquet(d_file)
    return data
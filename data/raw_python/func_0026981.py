def dropna_rows(data: pd.DataFrame, columns_name: str=None):
    """
    Remove columns with more NA values than threshold level

    :param data:
    :param columns_name:
    :return:

    """
    params = {}
    if columns_name is not None:
        params.update({'subset': columns_name.split(',')})
    data.dropna(inplace=True, **params)
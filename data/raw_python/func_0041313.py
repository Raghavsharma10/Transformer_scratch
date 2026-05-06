def index_row_dict_from_csv(path,
                            index_col=None,
                            iterator=False,
                            chunksize=None,
                            skiprows=None,
                            nrows=None,
                            use_ordered_dict=True,
                            **kwargs):
    """Read the csv into a dictionary. The key is it's index, the value
    is the dictionary form of the row.

    :param path: csv file path.
    :param index_col: None or str, the column that used as index.
    :param iterator:
    :param chunksize:
    :param skiprows:
    :param nrows:
    :param use_ordered_dict:

    :returns: {index_1: row1, index2: row2, ...} 

    **中文文档**

    读取csv, 选择一值完全不重复, 可作为index的列作为index, 生成一个字典
    数据结构, 使得可以通过index直接访问row。
    """
    _kwargs = dict(list(kwargs.items()))
    _kwargs["iterator"] = None
    _kwargs["chunksize"] = None
    _kwargs["skiprows"] = 0
    _kwargs["nrows"] = 1

    df = pd.read_csv(path, index_col=index_col, **_kwargs)
    columns = df.columns

    if index_col is None:
        raise Exception("please give index_col!")

    if use_ordered_dict:
        table = OrderedDict()
    else:
        table = dict()

    kwargs["iterator"] = iterator
    kwargs["chunksize"] = chunksize
    kwargs["skiprows"] = skiprows
    kwargs["nrows"] = nrows

    if iterator is True:
        for df in pd.read_csv(path, index_col=index_col, **kwargs):
            for ind, tp in zip(df.index, itertuple(df)):
                table[ind] = dict(zip(columns, tp))
    else:
        df = pd.read_csv(path, index_col=index_col, **kwargs)
        for ind, tp in zip(df.index, itertuple(df)):
            table[ind] = dict(zip(columns, tp))

    return table
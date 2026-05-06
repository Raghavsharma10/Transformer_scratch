def to_index_row_dict(df, index_col=None, use_ordered_dict=True):
    """Transform data frame to list of dict.

    :param index_col: None or str, the column that used as index.
    :param use_ordered_dict: if True, row dict is has same order as df.columns. 

    **中文文档**

    将dataframe以指定列为key, 转化成以行为视角的dict结构, 提升按行index访问
    的速度。若无指定列, 则使用index。
    """
    if index_col:
        index_list = df[index_col]
    else:
        index_list = df.index

    columns = df.columns

    if use_ordered_dict:
        table = OrderedDict()
    else:
        table = dict()

    for ind, tp in zip(index_list, itertuple(df)):
        table[ind] = dict(zip(columns, tp))

    return table
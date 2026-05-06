def to_dict_list(df, use_ordered_dict=True):
    """Transform each row to dict, and put them into a list.

    **中文文档**

    将 ``pandas.DataFrame`` 转换成一个字典的列表。列表的长度与行数相同, 其中
    每一个字典相当于表中的一行, 相当于一个 ``pandas.Series`` 对象。
    """
    if use_ordered_dict:
        dict = OrderedDict

    columns = df.columns
    data = list()
    for tp in itertuple(df):
        data.append(dict(zip(columns, tp)))
    return data
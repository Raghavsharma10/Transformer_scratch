def to_dict_list_generic_type(df, int_col=None, binary_col=None):
    """Transform each row to dict, and put them into a list. And automatically
    convert ``np.int64`` to ``int``, ``pandas.tslib.Timestamp`` to 
    ``datetime.datetime``, ``np.nan`` to ``None``.

    :param df: ``pandas.DataFrame`` instance.
    :param int_col: integer type columns.
    :param binary_col: binary type type columns.

    **中文文档**

    由于 ``pandas.Series`` 中的值的整数数据类型是 ``numpy.int64``, 
    时间数据类型是 ``pandas.tslib.Timestamp``, None的数据类型是 ``np.nan``。
    虽然从访问和计算的角度来说没有什么问题, 但会和很多数据库的操作不兼容。 

    此函数能将 ``pandas.DataFrame`` 转化成字典的列表。数据类型能正确的获得int, 
    bytes和datetime.datetime。
    """
    # Pre-process int_col, binary_col and datetime_col
    if (int_col is not None) and (not isinstance(int_col, (list, tuple))):
        int_col = [int_col, ]

    if (binary_col is not None) and (not isinstance(binary_col, (list, tuple))):
        binary_col = [binary_col, ]

    datetime_col = list()
    for col, dtype in dict(df.dtypes).items():
        if "datetime64" in str(dtype):
            datetime_col.append(col)
    if len(datetime_col) == 0:
        datetime_col = None

    # Pre-process binary column dataframe
    def b64_encode(b):
        try:
            return base64.b64encode(b)
        except:
            return b

    if binary_col is not None:
        for col in binary_col:
            df[col] = df[col].apply(b64_encode)

    data = json.loads(df.to_json(orient="records", date_format="iso"))

    if int_col is not None:
        for row in data:
            for col in int_col:
                try:
                    row[col] = int(row[col])
                except:
                    pass

    if binary_col is not None:
        for row in data:
            for col in binary_col:
                try:
                    row[col] = base64.b64decode(row[col].encode("ascii"))
                except:
                    pass

    if datetime_col is not None:
        for row in data:
            for col in datetime_col:
                try:
                    row[col] = rolex.str2datetime(row[col])
                except:
                    pass

    return data
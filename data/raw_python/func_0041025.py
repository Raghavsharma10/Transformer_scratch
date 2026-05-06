def insert_data_frame(col, df, int_col=None, binary_col=None, minimal_size=5):
    """Insert ``pandas.DataFrame``.

    :param col: :class:`pymongo.collection.Collection` instance.
    :param df: :class:`pandas.DataFrame` instance.
    :param int_col: list of integer-type column.
    :param binary_col: list of binary-type column.
    """
    data = transform.to_dict_list_generic_type(df,
                                               int_col=int_col,
                                               binary_col=binary_col)
    smart_insert(col, data, minimal_size)
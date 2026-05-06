def grouper_df(df, chunksize):
    """Evenly divide pd.DataFrame into n rows piece, no filled value 
    if sub dataframe's size smaller than n.

    :param df: ``pandas.DataFrame`` instance.
    :param chunksize: number of rows of each small DataFrame.

    **中文文档**

    将 ``pandas.DataFrame`` 分拆成等大小的小DataFrame。
    """
    data = list()
    counter = 0
    for tp in zip(*(l for col, l in df.iteritems())):
        counter += 1
        data.append(tp)
        if counter == chunksize:
            new_df = pd.DataFrame(data, columns=df.columns)
            yield new_df
            data = list()
            counter = 0

    if len(data) > 0:
        new_df = pd.DataFrame(data, columns=df.columns)
        yield new_df
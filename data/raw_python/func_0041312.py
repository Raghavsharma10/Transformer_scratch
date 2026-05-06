def iter_tuple_from_csv(path,
                        iterator=False,
                        chunksize=None,
                        skiprows=None,
                        nrows=None,
                        **kwargs):
    """A high performance, low memory usage csv file row iterator function.

    :param path: csv file path.
    :param iterator:
    :param chunksize:
    :param skiprows:
    :param nrows:

    :yield tuple: 

    **中文文档**

    对dataframe进行tuple风格的高性能行遍历。

    对用pandas从csv文件读取的dataframe进行逐行遍历时, iterrows和itertuple
    都不是性能最高的方法。这是因为iterrows要生成Series对象, 而itertuple
    也要对index进行访问。所以本方法是使用内建zip方法对所有的column进行打包
    解压, 所以性能上是最佳的。
    """
    kwargs["iterator"] = iterator
    kwargs["chunksize"] = chunksize
    kwargs["skiprows"] = skiprows
    kwargs["nrows"] = nrows

    if iterator is True:
        for df in pd.read_csv(path, **kwargs):
            for tp in itertuple(df):
                yield tp
    else:
        df = pd.read_csv(path, **kwargs)
        for tp in itertuple(df):
            yield tp
def dump(df,fp):
    """
    dump DataFrame to file
    :param DataFrame df: 
    :param file fp: 
    """
    arff = __dump(df)
    liacarff.dump(arff,fp)
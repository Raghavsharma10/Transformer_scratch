def __load(arff):
    """
    load liac-arff to pandas DataFrame
    :param dict arff:arff dict created liac-arff
    :rtype: DataFrame
    :return: pandas DataFrame
    """
    attrs = arff['attributes']
    attrs_t = []
    for attr in attrs:
        if isinstance(attr[1], list):
            attrs_t.append("%s@{%s}" % (attr[0], ','.join(attr[1])))
        else:
            attrs_t.append("%s@%s" % (attr[0], attr[1]))

    df = pd.DataFrame(data=arff['data'], columns=attrs_t)
    return df
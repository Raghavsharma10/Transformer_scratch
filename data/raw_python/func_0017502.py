def __dump(df,relation='data',description=''):
    """
    dump DataFrame to liac-arff
    :param DataFrame df: 
    :param str relation: 
    :param str description: 
    :rtype: dict
    :return: liac-arff dict 
    """
    attrs = []
    for col in df.columns:
        attr = col.split('@')
        if attr[1].count('{')>0 and attr[1].count('}')>0:
            vals = attr[1].replace('{','').replace('}','').split(',')
            attrs.append((attr[0],vals))
        else:
            attrs.append((attr[0],attr[1]))

    data = list(df.values)
    result = {
        'attributes':attrs,
        'data':data,
        'description':description,
        'relation':relation
    }
    return result
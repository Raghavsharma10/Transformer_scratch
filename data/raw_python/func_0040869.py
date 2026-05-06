def result_to_dict(res):
    """
    :param res: :any:`sqlalchemy.engine.ResultProxy`

    :return: a list of dicts where each dict represents a row in the query where the key \
    is the column name and the value is the value of that column.
    """
    keys = res.keys()
    return [dict(itertools.izip(keys, row)) for row in res]
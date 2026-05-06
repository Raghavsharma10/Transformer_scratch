def union_sql(view_name, *tables):
    """This function generates string containing SQL code, that creates
    a big VIEW, that consists of many SELECTs.

    >>> utils.union_sql('global', 'foo', 'bar', 'baz')
    'CREATE VIEW global SELECT * FROM foo UNION SELECT * FROM bar UNION SELECT * FROM baz'
    """

    if not tables:
        raise Exception("no tables given")

    ret = ""
    pre = "CREATE VIEW %s AS SELECT * FROM " % view_name

    for table in tables:
        ret += pre + table
        pre = " UNION SELECT * FROM "

    return ret
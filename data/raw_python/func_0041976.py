def makeCoreValuesSubqueryCondition(engine, column, values: List[Union[int, str]]):
    """ Make Core Values Subquery

    :param engine: The database engine, used to determine the dialect
    :param column: The column, eg TableItem.__table__.c.colName
    :param values: A list of string or int values
    """

    if isPostGreSQLDialect(engine):
        return column.in_(values)

    if not isMssqlDialect(engine):
        raise NotImplementedError()

    sql = _createMssqlSqlText(values)

    return column.in_(sql)
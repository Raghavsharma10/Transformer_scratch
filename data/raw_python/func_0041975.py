def makeOrmValuesSubqueryCondition(ormSession, column, values: List[Union[int, str]]):
    """ Make Orm Values Subquery

    :param ormSession: The orm session instance
    :param column: The column from the Declarative table, eg TableItem.colName
    :param values: A list of string or int values
    """
    if isPostGreSQLDialect(ormSession.bind):
        return column.in_(values)

    if not isMssqlDialect(ormSession.bind):
        raise NotImplementedError()

    sql = _createMssqlSqlText(values)

    sub_qry = ormSession.query(column)  # Any column, it just assigns a name
    sub_qry = sub_qry.from_statement(sql)

    return column.in_(sub_qry)
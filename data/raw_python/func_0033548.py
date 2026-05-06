def parse(table, query=None, date=None, fields=None,
          distinct=False, limit=None, alias=None):
    '''
    Given a SQLAlchemy Table() instance, generate a SQLAlchemy
    Query() instance with the given parameters.

    :param table: SQLAlchemy Table() instance
    :param query: MQL query
    :param date: metrique date range query
    :param date: metrique date range query element
    :param fields: list of field names to return as columns
    :param distinct: apply DISTINCT to this query
    :param limit: apply LIMIT to this query
    :param alias: apply ALIAS AS to this query
    '''
    date = date_range(date)
    limit = int(limit or -1)
    if query and date:
        query = '%s and %s' % (query, date)
    elif date:
        query = date
    elif query:
        pass
    else:  # date is null, query is not
        query = None

    fields = parse_fields(fields=fields) or None
    # we must pass in the table column objects themselves to ensure
    # our bind / result processors are mapped properly
    fields = fields if fields else table.columns

    msg = 'parse(query=%s, fields=%s)' % (query, fields)
    #msg = re.sub(' in \[[^\]]+\]', ' in [...]', msg)
    logger.debug(msg)
    kwargs = {}
    if query:
        interpreter = MQLInterpreter(table)
        query = interpreter.parse(query)
        kwargs['whereclause'] = query
    if distinct:
        kwargs['distinct'] = distinct
    query = select(fields, from_obj=table, **kwargs)
    if limit >= 1:
        query = query.limit(limit)
    if alias:
        query = query.alias(alias)
    return query
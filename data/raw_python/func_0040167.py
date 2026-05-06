def get(
    table,
    session,
    version_id=None,
    t1=None,
    t2=None,
    fields=None,
    conds=None,
    include_deleted=True,
    page=1,
    page_size=100,
):
    """
    :param table: the model class which inherits from
        :class:`~savage.models.user_table.SavageModelMixin` and specifies the model of
        the user table from which we are querying
    :param session: a sqlalchemy session with connections to the database
    :param version_id: if specified, the value of t1 and t2 will be ignored. If specified, this will
        return all records after the specified version_id.
    :param t1: lower bound time for this query; if None or unspecified,
        defaults to the unix epoch. If this is specified and t2 is not, this query
        will simply return the time slice of data at t1. This must either be a valid
        sql time string or a datetime.datetime object.
    :param t2: upper bound time for this query; if both t1 and t2 are none or unspecified,
        this will return the latest data (i.e. time slice of data now). This must either be a
        valid sql time string or a datetime.datetime object.
    :param fields: a list of strings which corresponds to columns in the table; If
        None or unspecified, returns all fields in the table.
    :param conds: a list of dictionary of key value pairs where keys are columns in the table
        and values are values the column should take on. If specified, this query will
        only return rows where the columns meet all the conditions. The columns specified
        in this dictionary must be exactly the unique columns that versioning pivots around.
    :param include_deleted: if ``True``, the response will include deleted changes. Else it will
        only include changes where ``deleted = 0`` i.e. the data was in the user table.
    :param page: the offset of the result set (1-indexed); i.e. if page_size is 100 and page is 2,
        the result set will contain results 100 - 199
    :param page_size: upper bound on number of results to display. Note the actual returned result
        set may be smaller than this due to the roll up.
    """
    limit, offset = _get_limit_and_offset(page, page_size)
    version_col_names = table.version_columns
    if fields is None:
        fields = [name for name in utils.get_column_names(table) if name != 'version_id']

    if version_id is not None:
        return _format_response(utils.result_to_dict(session.execute(
            sa.select([table.ArchiveTable])
            .where(table.ArchiveTable.version_id > version_id)
            .order_by(*_get_order_clause(table.ArchiveTable))
            .limit(page_size)
            .offset(offset)
        )), fields, version_col_names)

    if t1 is None and t2 is None:
        rows = _get_latest_time_slice(table, session, conds, include_deleted, limit, offset)
        return _format_response(rows, fields, version_col_names)

    if t2 is None:  # return a historical time slice
        rows = _get_historical_time_slice(
            table, session, t1, conds, include_deleted, limit, offset
        )
        return _format_response(rows, fields, version_col_names)

    if t1 is None:
        t1 = datetime.utcfromtimestamp(0)

    rows = _get_historical_changes(
        table, session, conds, t1, t2, include_deleted, limit, offset
    )
    return _format_response(rows, fields, version_col_names)
def get_tags(user):
    """Get all tags."""
    args = schemas.args(flask.request.args.to_dict())
    query = v1_utils.QueryBuilder(_TABLE, args, _T_COLUMNS)
    nb_rows = query.get_number_of_rows()
    rows = query.execute(fetchall=True)
    rows = v1_utils.format_result(rows, _TABLE.name)
    return flask.jsonify({'tags': rows, '_meta': {'count': nb_rows}})
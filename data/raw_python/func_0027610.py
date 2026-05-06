def get_all_components(user, topic_id):
    """Get all components of a topic."""

    args = schemas.args(flask.request.args.to_dict())

    query = v1_utils.QueryBuilder(_TABLE, args, _C_COLUMNS)

    query.add_extra_condition(sql.and_(
        _TABLE.c.topic_id == topic_id,
        _TABLE.c.state != 'archived'))

    nb_rows = query.get_number_of_rows()
    rows = query.execute(fetchall=True)
    rows = v1_utils.format_result(rows, _TABLE.name, args['embed'],
                                  _EMBED_MANY)

    # Return only the component which have the export_control flag set to true
    #
    if user.is_not_super_admin():
        rows = [row for row in rows if row['export_control']]

    return flask.jsonify({'components': rows, '_meta': {'count': nb_rows}})
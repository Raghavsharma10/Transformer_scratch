def purge_archived_resources(user, table):
    """Remove the entries to be purged from the database. """

    if user.is_not_super_admin():
        raise dci_exc.Unauthorized()

    where_clause = sql.and_(
        table.c.state == 'archived'
    )
    query = table.delete().where(where_clause)
    flask.g.db_conn.execute(query)

    return flask.Response(None, 204, content_type='application/json')
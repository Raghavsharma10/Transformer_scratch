def refresh_api_secret(user, resource, table):
    """Refresh the resource API Secret. """

    resource_name = table.name[0:-1]

    where_clause = sql.and_(
        table.c.etag == resource['etag'],
        table.c.id == resource['id'],
    )

    values = {
        'api_secret': signature.gen_secret(),
        'etag': utils.gen_etag()
    }

    query = table.update().where(where_clause).values(**values)
    result = flask.g.db_conn.execute(query)

    if not result.rowcount:
        raise dci_exc.DCIConflict(resource_name, resource['id'])

    res = flask.jsonify(({'id': resource['id'], 'etag': resource['etag'],
                          'api_secret': values['api_secret']}))
    res.headers.add_header('ETag', values['etag'])
    return res
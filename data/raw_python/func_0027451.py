def verify_existence_and_get(id, table, name=None, get_id=False):
    """Verify the existence of a resource in the database and then
    return it if it exists, according to the condition, or raise an
    exception.

    :param id: id of the resource
    :param table: the table object
    :param name: the name of the row to look for
    :param get_id: if True, return only the ID
    :return:
    """

    where_clause = table.c.id == id
    if name:
        where_clause = table.c.name == name

    if 'state' in table.columns:
        where_clause = sql.and_(table.c.state != 'archived', where_clause)

    query = sql.select([table]).where(where_clause)
    result = flask.g.db_conn.execute(query).fetchone()

    if result is None:
        raise dci_exc.DCIException('Resource "%s" not found.' % id,
                                   status_code=404)
    if get_id:
        return result.id
    return result
def retrieve_tags_from_component(user, c_id):
    """Retrieve all tags attached to a component."""
    JCT = models.JOIN_COMPONENTS_TAGS
    query = (sql.select([models.TAGS])
             .select_from(JCT.join(models.TAGS))
             .where(JCT.c.component_id == c_id))
    rows = flask.g.db_conn.execute(query)

    return flask.jsonify({'tags': rows, '_meta': {'count': rows.rowcount}})
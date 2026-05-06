def verify_and_get_components_ids(topic_id, components_ids, component_types,
                                  db_conn=None):
    """Process some verifications of the provided components ids."""
    db_conn = db_conn or flask.g.db_conn
    if len(components_ids) != len(component_types):
        msg = 'The number of component ids does not match the number ' \
              'of component types %s' % component_types
        raise dci_exc.DCIException(msg, status_code=412)

    # get the components from their ids
    schedule_component_types = set()
    for c_id in components_ids:
        where_clause = sql.and_(models.COMPONENTS.c.id == c_id,
                                models.COMPONENTS.c.topic_id == topic_id,
                                models.COMPONENTS.c.export_control == True,  # noqa
                                models.COMPONENTS.c.state == 'active')
        query = (sql.select([models.COMPONENTS])
                 .where(where_clause))
        cmpt = db_conn.execute(query).fetchone()

        if cmpt is None:
            msg = 'Component id %s not found or not exported' % c_id
            raise dci_exc.DCIException(msg, status_code=412)
        cmpt = dict(cmpt)

        if cmpt['type'] in schedule_component_types:
            msg = ('Component types malformed: type %s duplicated.' %
                   cmpt['type'])
            raise dci_exc.DCIException(msg, status_code=412)
        schedule_component_types.add(cmpt['type'])
    return components_ids
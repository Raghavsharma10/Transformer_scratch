def get_last_components_by_type(component_types, topic_id, db_conn=None):
    """For each component type of a topic, get the last one."""
    db_conn = db_conn or flask.g.db_conn
    schedule_components_ids = []
    for ct in component_types:
        where_clause = sql.and_(models.COMPONENTS.c.type == ct,
                                models.COMPONENTS.c.topic_id == topic_id,
                                models.COMPONENTS.c.export_control == True,
                                models.COMPONENTS.c.state == 'active')  # noqa
        query = (sql.select([models.COMPONENTS.c.id])
                 .where(where_clause)
                 .order_by(sql.desc(models.COMPONENTS.c.created_at)))
        cmpt_id = db_conn.execute(query).fetchone()

        if cmpt_id is None:
            msg = 'Component of type "%s" not found or not exported.' % ct
            raise dci_exc.DCIException(msg, status_code=412)

        cmpt_id = cmpt_id[0]
        if cmpt_id in schedule_components_ids:
            msg = ('Component types %s malformed: type %s duplicated.' %
                   (component_types, ct))
            raise dci_exc.DCIException(msg, status_code=412)
        schedule_components_ids.append(cmpt_id)
    return schedule_components_ids
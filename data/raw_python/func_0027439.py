def get_remoteci_configuration(topic_id, remoteci_id, db_conn=None):
    """Get a remoteci configuration. This will iterate over each
    configuration in a round robin manner depending on the last
    rconfiguration used by the remoteci."""

    db_conn = db_conn or flask.g.db_conn
    last_rconfiguration_id = get_last_rconfiguration_id(
        topic_id, remoteci_id, db_conn=db_conn)
    _RCONFIGURATIONS = models.REMOTECIS_RCONFIGURATIONS
    _J_RCONFIGURATIONS = models.JOIN_REMOTECIS_RCONFIGURATIONS
    query = sql.select([_RCONFIGURATIONS]). \
        select_from(_J_RCONFIGURATIONS.
                    join(_RCONFIGURATIONS)). \
        where(_J_RCONFIGURATIONS.c.remoteci_id == remoteci_id)
    query = query.where(sql.and_(_RCONFIGURATIONS.c.state != 'archived',
                                 _RCONFIGURATIONS.c.topic_id == topic_id))
    query = query.order_by(sql.desc(_RCONFIGURATIONS.c.created_at))
    query = query.order_by(sql.asc(_RCONFIGURATIONS.c.name))
    all_rconfigurations = db_conn.execute(query).fetchall()

    if len(all_rconfigurations) > 0:
        for i in range(len(all_rconfigurations)):
            if str(all_rconfigurations[i]['id']) == last_rconfiguration_id:
                # if i==0, then indice -1 is the last element
                return all_rconfigurations[i - 1]
        return all_rconfigurations[0]
    else:
        return None
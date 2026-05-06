def get_last_rconfiguration_id(topic_id, remoteci_id, db_conn=None):
    """Get the rconfiguration_id of the last job run by the remoteci.

    :param topic_id: the topic
    :param remoteci_id: the remoteci id
    :return: last rconfiguration_id of the remoteci
    """
    db_conn = db_conn or flask.g.db_conn
    __TABLE = models.JOBS
    query = sql.select([__TABLE.c.rconfiguration_id]). \
        order_by(sql.desc(__TABLE.c.created_at)). \
        where(sql.and_(__TABLE.c.topic_id == topic_id,
                       __TABLE.c.remoteci_id == remoteci_id)). \
        limit(1)
    rconfiguration_id = db_conn.execute(query).fetchone()
    if rconfiguration_id is not None:
        return str(rconfiguration_id[0])
    else:
        return None
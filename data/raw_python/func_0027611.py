def get_component_types_from_topic(topic_id, db_conn=None):
    """Returns the component types of a topic."""
    db_conn = db_conn or flask.g.db_conn
    query = sql.select([models.TOPICS]).\
        where(models.TOPICS.c.id == topic_id)
    topic = db_conn.execute(query).fetchone()
    topic = dict(topic)
    return topic['component_types']
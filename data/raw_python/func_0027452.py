def user_topic_ids(user):
    """Retrieve the list of topics IDs a user has access to."""

    if user.is_super_admin() or user.is_read_only_user():
        query = sql.select([models.TOPICS])
    else:
        query = (sql.select([models.JOINS_TOPICS_TEAMS.c.topic_id])
                 .select_from(
                     models.JOINS_TOPICS_TEAMS.join(
                         models.TOPICS, sql.and_(models.JOINS_TOPICS_TEAMS.c.topic_id == models.TOPICS.c.id,  # noqa
                                                 models.TOPICS.c.state == 'active'))  # noqa
                  ).where(
                      sql.or_(models.JOINS_TOPICS_TEAMS.c.team_id.in_(user.teams_ids),  # noqa
                              models.JOINS_TOPICS_TEAMS.c.team_id.in_(user.child_teams_ids))))  # noqa

    rows = flask.g.db_conn.execute(query).fetchall()
    return [str(row[0]) for row in rows]
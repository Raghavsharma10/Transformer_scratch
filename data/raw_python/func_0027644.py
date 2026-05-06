def get_all_jobs(user, topic_id=None):
    """Get all jobs.

    If topic_id is not None, then return all the jobs with a topic
    pointed by topic_id.
    """
    # get the diverse parameters
    args = schemas.args(flask.request.args.to_dict())

    # build the query thanks to the QueryBuilder class
    query = v1_utils.QueryBuilder(_TABLE, args, _JOBS_COLUMNS)

    # add extra conditions for filtering

    # # If not admin nor rh employee then restrict the view to the team
    if user.is_not_super_admin() and not user.is_read_only_user():
        query.add_extra_condition(
            sql.or_(
                _TABLE.c.team_id.in_(user.teams_ids),
                _TABLE.c.team_id.in_(user.child_teams_ids)))

    # # If topic_id not None, then filter by topic_id
    if topic_id is not None:
        query.add_extra_condition(_TABLE.c.topic_id == topic_id)

    # # Get only the non archived jobs
    query.add_extra_condition(_TABLE.c.state != 'archived')

    nb_rows = query.get_number_of_rows()
    rows = query.execute(fetchall=True)
    rows = v1_utils.format_result(rows, _TABLE.name, args['embed'],
                                  _EMBED_MANY)

    return flask.jsonify({'jobs': rows, '_meta': {'count': nb_rows}})
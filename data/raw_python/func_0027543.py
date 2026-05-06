def get_all_analytics(user, job_id):
    """Get all analytics of a job."""

    args = schemas.args(flask.request.args.to_dict())
    v1_utils.verify_existence_and_get(job_id, models.JOBS)

    query = v1_utils.QueryBuilder(_TABLE, args, _A_COLUMNS)
    # If not admin nor rh employee then restrict the view to the team
    if user.is_not_super_admin() and not user.is_read_only_user():
        query.add_extra_condition(_TABLE.c.team_id.in_(user.teams_ids))

    query.add_extra_condition(_TABLE.c.job_id == job_id)

    nb_rows = query.get_number_of_rows()
    rows = query.execute(fetchall=True)
    rows = v1_utils.format_result(rows, _TABLE.name)

    return flask.jsonify({'analytics': rows, '_meta': {'count': nb_rows}})
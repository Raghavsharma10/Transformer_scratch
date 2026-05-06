def get_all_jobstates(user, job_id):
    """Get all jobstates.
    """
    args = schemas.args(flask.request.args.to_dict())
    job = v1_utils.verify_existence_and_get(job_id, models.JOBS)
    if user.is_not_super_admin() and user.is_not_read_only_user():
        if (job['team_id'] not in user.teams_ids and
            job['team_id'] not in user.child_teams_ids):
            raise dci_exc.Unauthorized()

    query = v1_utils.QueryBuilder(_TABLE, args, _JS_COLUMNS)
    query.add_extra_condition(_TABLE.c.job_id == job_id)

    # get the number of rows for the '_meta' section
    nb_rows = query.get_number_of_rows()
    rows = query.execute(fetchall=True)
    rows = v1_utils.format_result(rows, _TABLE.name, args['embed'],
                                  _EMBED_MANY)
    return flask.jsonify({'jobstates': rows, '_meta': {'count': nb_rows}})
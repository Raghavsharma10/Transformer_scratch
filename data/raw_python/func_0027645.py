def update_job_by_id(user, job_id):
    """Update a job
    """
    # get If-Match header
    if_match_etag = utils.check_and_get_etag(flask.request.headers)

    # get the diverse parameters
    values = schemas.job.put(flask.request.json)

    job = v1_utils.verify_existence_and_get(job_id, _TABLE)
    job = dict(job)

    if not user.is_in_team(job['team_id']):
        raise dci_exc.Unauthorized()

    # Update jobstate if needed
    status = values.get('status')
    if status and job.get('status') != status:
        jobstates.insert_jobstate(user, {
            'status': status,
            'job_id': job_id
        })
        if status in models.FINAL_STATUSES:
            jobs_events.create_event(job_id, status, job['topic_id'])

    where_clause = sql.and_(_TABLE.c.etag == if_match_etag,
                            _TABLE.c.id == job_id)

    values['etag'] = utils.gen_etag()
    query = _TABLE.update().returning(*_TABLE.columns).\
        where(where_clause).values(**values)

    result = flask.g.db_conn.execute(query)
    if not result.rowcount:
        raise dci_exc.DCIConflict('Job', job_id)

    return flask.Response(
        json.dumps({'job': result.fetchone()}), 200,
        headers={'ETag': values['etag']},
        content_type='application/json'
    )
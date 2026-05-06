def delete_tag_from_job(user, job_id, tag_id):
    """Delete a tag from a job."""

    _JJT = models.JOIN_JOBS_TAGS
    job = v1_utils.verify_existence_and_get(job_id, _TABLE)
    if not user.is_in_team(job['team_id']):
        raise dci_exc.Unauthorized()
    v1_utils.verify_existence_and_get(tag_id, models.TAGS)

    query = _JJT.delete().where(sql.and_(_JJT.c.tag_id == tag_id,
                                         _JJT.c.job_id == job_id))

    try:
        flask.g.db_conn.execute(query)
    except sa_exc.IntegrityError:
        raise dci_exc.DCICreationConflict('tag', 'tag_id')

    return flask.Response(None, 204, content_type='application/json')
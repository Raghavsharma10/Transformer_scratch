def get_tags_from_job(user, job_id):
    """Retrieve all tags attached to a job."""

    job = v1_utils.verify_existence_and_get(job_id, _TABLE)
    if not user.is_in_team(job['team_id']) and not user.is_read_only_user():
        raise dci_exc.Unauthorized()

    JTT = models.JOIN_JOBS_TAGS
    query = (sql.select([models.TAGS])
             .select_from(JTT.join(models.TAGS))
             .where(JTT.c.job_id == job_id))
    rows = flask.g.db_conn.execute(query)

    return flask.jsonify({'tags': rows, '_meta': {'count': rows.rowcount}})
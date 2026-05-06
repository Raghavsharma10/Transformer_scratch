def get_analytic(user, job_id, anc_id):
    """Get an analytic."""

    v1_utils.verify_existence_and_get(job_id, models.JOBS)
    analytic = v1_utils.verify_existence_and_get(anc_id, _TABLE)
    analytic = dict(analytic)
    if not user.is_in_team(analytic['team_id']):
        raise dci_exc.Unauthorized()
    return flask.jsonify({'analytic': analytic})
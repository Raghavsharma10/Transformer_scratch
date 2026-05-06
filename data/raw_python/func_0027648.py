def add_tag_to_job(user, job_id):
    """Add a tag to a job."""

    job = v1_utils.verify_existence_and_get(job_id, _TABLE)
    if not user.is_in_team(job['team_id']):
        raise dci_exc.Unauthorized()

    values = {
        'job_id': job_id
    }

    job_tagged = tags.add_tag_to_resource(values, models.JOIN_JOBS_TAGS)

    return flask.Response(json.dumps(job_tagged), 201,
                          content_type='application/json')
def create_new_update_job_from_an_existing_job(user, job_id):
    """Create a new job in the same topic as the job_id provided and
    associate the latest components of this topic."""

    values = {
        'id': utils.gen_uuid(),
        'created_at': datetime.datetime.utcnow().isoformat(),
        'updated_at': datetime.datetime.utcnow().isoformat(),
        'etag': utils.gen_etag(),
        'status': 'new'
    }

    original_job_id = job_id
    original_job = v1_utils.verify_existence_and_get(original_job_id,
                                                     models.JOBS)

    if not user.is_in_team(original_job['team_id']):
        raise dci_exc.Unauthorized()

    # get the remoteci
    remoteci_id = str(original_job['remoteci_id'])
    remoteci = v1_utils.verify_existence_and_get(remoteci_id,
                                                 models.REMOTECIS)
    values.update({'remoteci_id': remoteci_id})

    # get the associated topic
    topic_id = str(original_job['topic_id'])
    v1_utils.verify_existence_and_get(topic_id, models.TOPICS)

    values.update({
        'user_agent': flask.request.environ.get('HTTP_USER_AGENT'),
        'client_version': flask.request.environ.get(
            'HTTP_CLIENT_VERSION'
        ),
    })

    values = _build_job(topic_id, remoteci, [], values,
                        update_previous_job_id=original_job_id)

    return flask.Response(json.dumps({'job': values}), 201,
                          headers={'ETag': values['etag']},
                          content_type='application/json')
def create_new_upgrade_job_from_an_existing_job(user):
    """Create a new job in the 'next topic' of the topic of
    the provided job_id."""
    values = schemas.job_upgrade.post(flask.request.json)

    values.update({
        'id': utils.gen_uuid(),
        'created_at': datetime.datetime.utcnow().isoformat(),
        'updated_at': datetime.datetime.utcnow().isoformat(),
        'etag': utils.gen_etag(),
        'status': 'new'
    })

    original_job_id = values.pop('job_id')
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
    topic = v1_utils.verify_existence_and_get(topic_id, models.TOPICS)

    values.update({
        'user_agent': flask.request.environ.get('HTTP_USER_AGENT'),
        'client_version': flask.request.environ.get(
            'HTTP_CLIENT_VERSION'
        ),
    })

    next_topic_id = topic['next_topic_id']

    if not next_topic_id:
        raise dci_exc.DCIException(
            "topic %s does not contains a next topic" % topic_id)

    # instantiate a new job in the next_topic_id
    # todo(yassine): make possible the upgrade to choose specific components
    values = _build_job(next_topic_id, remoteci, [], values,
                        previous_job_id=original_job_id)

    return flask.Response(json.dumps({'job': values}), 201,
                          headers={'ETag': values['etag']},
                          content_type='application/json')
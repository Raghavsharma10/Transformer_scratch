def schedule_jobs(user):
    """Dispatch jobs to remotecis.

    The remoteci can use this method to request a new job.

    Before a job is dispatched, the server will flag as 'killed' all the
    running jobs that were associated with the remoteci. This is because they
    will never be finished.
    """

    values = schemas.job_schedule.post(flask.request.json)

    values.update({
        'id': utils.gen_uuid(),
        'created_at': datetime.datetime.utcnow().isoformat(),
        'updated_at': datetime.datetime.utcnow().isoformat(),
        'etag': utils.gen_etag(),
        'status': 'new',
        'remoteci_id': user.id,
        'user_agent': flask.request.environ.get('HTTP_USER_AGENT'),
        'client_version': flask.request.environ.get(
            'HTTP_CLIENT_VERSION'
        ),
    })

    topic_id = values.pop('topic_id')
    topic_id_secondary = values.pop('topic_id_secondary')
    components_ids = values.pop('components_ids')

    # check remoteci
    remoteci = v1_utils.verify_existence_and_get(user.id, models.REMOTECIS)
    if remoteci['state'] != 'active':
        message = 'RemoteCI "%s" is disabled.' % remoteci['id']
        raise dci_exc.DCIException(message, status_code=412)

    # check primary topic
    topic = v1_utils.verify_existence_and_get(topic_id, models.TOPICS)
    if topic['state'] != 'active':
        msg = 'Topic %s:%s not active.' % (topic_id, topic['name'])
        raise dci_exc.DCIException(msg, status_code=412)
    v1_utils.verify_team_in_topic(user, topic_id)

    # check secondary topic
    if topic_id_secondary:
        topic_secondary = v1_utils.verify_existence_and_get(
            topic_id_secondary, models.TOPICS)
        if topic_secondary['state'] != 'active':
            msg = 'Topic %s:%s not active.' % (topic_id_secondary,
                                               topic['name'])
            raise dci_exc.DCIException(msg, status_code=412)
        v1_utils.verify_team_in_topic(user, topic_id_secondary)

    dry_run = values.pop('dry_run')
    if dry_run:
        component_types = components.get_component_types_from_topic(topic_id)
        components_ids = components.get_last_components_by_type(
            component_types,
            topic_id
        )
        return flask.Response(
            json.dumps({'components_ids': components_ids, 'job': None}),
            201,
            content_type='application/json'
        )

    remotecis.kill_existing_jobs(remoteci['id'])

    values = _build_job(topic_id, remoteci, components_ids, values,
                        topic_id_secondary=topic_id_secondary)

    return flask.Response(json.dumps({'job': values}), 201,
                          headers={'ETag': values['etag']},
                          content_type='application/json')
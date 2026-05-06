def attach_issue(resource_id, table, user_id):
    """Attach an issue to a specific job."""
    data = schemas.issue.post(flask.request.json)
    issue = _get_or_create_issue(data)

    # Second, insert a join record in the JOIN_JOBS_ISSUES or
    # JOIN_COMPONENTS_ISSUES database.
    if table.name == 'jobs':
        join_table = models.JOIN_JOBS_ISSUES
    else:
        join_table = models.JOIN_COMPONENTS_ISSUES

    key = '%s_id' % table.name[0:-1]
    query = join_table.insert().values({
        'user_id': user_id,
        'issue_id': issue['id'],
        key: resource_id
    })

    try:
        flask.g.db_conn.execute(query)
    except sa_exc.IntegrityError:
        raise dci_exc.DCICreationConflict(join_table.name,
                                          '%s, issue_id' % key)

    result = json.dumps({'issue': dict(issue)})
    return flask.Response(result, 201, content_type='application/json')
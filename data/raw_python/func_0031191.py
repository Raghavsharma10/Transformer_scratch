def delete_instance(credentials, project, zone, name, wait_until_done=False):
    """Delete an instance.

    TODO: docstring
    """

    access_token = credentials.get_access_token()

    headers = {
        'Authorization': 'Bearer %s' % access_token.access_token
    }

    r = requests.delete('https://www.googleapis.com/compute/v1/'
                        'projects/%s/zones/%s/instances/%s'
                        % (project, zone, name),
                        headers=headers)

    r.raise_for_status()

    op_name = r.json()['name']

    _LOGGER.info('Submitted request to create intsance '
                '(HTTP code: %d).',
                r.status_code)

    if wait_until_done:
        _LOGGER.info('Waiting until operation is done...')
        wait_for_zone_op(access_token, project, zone, op_name)

    return op_name
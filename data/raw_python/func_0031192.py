def wait_for_instance_deletion(credentials, project, zone, instance_name,
                               interval_seconds=5):
    """Wait until an instance is deleted.
    
    We require that initially, the specified instance exists.
    TODO: docstring
    """
    
    t0 = time.time()
    access_token = credentials.get_access_token()
    headers = {
        'Authorization': 'Bearer %s' % access_token.access_token
    }

    r = requests.get('https://www.googleapis.com/compute/v1/'
                     'projects/%s/zones/%s/instances/%s'
                     % (project, zone, instance_name),
                     headers=headers)
    
    if r.status_code == 404:
        raise AssertionError('Instance "%s" does not exist!' % instance_name)
        
    r.raise_for_status()
    _LOGGER.debug('Instance "%s" exists.', instance_name)

    while True:
        time.sleep(interval_seconds)

        access_token = credentials.get_access_token()
        headers = {
            'Authorization': 'Bearer %s' % access_token.access_token
        }

        r = requests.get('https://www.googleapis.com/compute/v1/'
                         'projects/%s/zones/%s/instances/%s'
                         % (project, zone, instance_name),
                         headers=headers)
        if r.status_code == 404:
            break
        r.raise_for_status()
        _LOGGER.debug('Instance "%s" still exists.', instance_name)
        
    t1 = time.time()
    t = t1-t0
    t_min = t/60.0
    _LOGGER.info('Instance was deleted after %.1f s (%.1f m).', t, t_min)
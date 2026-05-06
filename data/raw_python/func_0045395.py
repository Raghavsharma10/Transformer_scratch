def _remote_status(session, service_id, uuid, url, interval=3):
    """Poll for remote command status."""
    _LOGGER.info('polling for status')
    resp = session.get(url, params={
        'remoteServiceRequestID':service_id,
        'uuid':uuid
    }).json()
    if resp['status'] == 'SUCCESS':
        return 'completed'
    time.sleep(interval)
    return _remote_status(session, service_id, uuid, url)
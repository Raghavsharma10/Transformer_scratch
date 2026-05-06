def query_api(endpoint, log):
    """Query the AppVeyor API.

    :raise HandledError: On non HTTP200 responses or invalid JSON response.

    :param str endpoint: API endpoint to query (e.g. '/projects/Robpol86/appveyor-artifacts').
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.

    :return: Parsed JSON response.
    :rtype: dict
    """
    url = API_PREFIX + endpoint
    headers = {'content-type': 'application/json'}
    response = None
    log.debug('Querying %s with headers %s.', url, headers)
    for i in range(QUERY_ATTEMPTS):
        try:
            try:
                response = requests.get(url, headers=headers, timeout=10)
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.Timeout):
                log.error('Timed out waiting for reply from server.')
                raise HandledError
            except requests.ConnectionError:
                log.error('Unable to connect to server.')
                raise HandledError
        except HandledError:
            if i == QUERY_ATTEMPTS - 1:
                raise
            log.warning('Network error, retrying in 1 second...')
            time.sleep(1)
        else:
            break
    log.debug('Response status: %d', response.status_code)
    log.debug('Response headers: %s', str(response.headers))
    log.debug('Response text: %s', response.text)

    if not response.ok:
        message = response.json().get('message')
        if message:
            log.error('HTTP %d: %s', response.status_code, message)
        else:
            log.error('HTTP %d: Unknown error: %s', response.status_code, response.text)
        raise HandledError

    try:
        return response.json()
    except ValueError:
        log.error('Failed to parse JSON: %s', response.text)
        raise HandledError
def _get_from_dapi_or_mirror(link):
    '''Tries to get the link form DAPI or the mirror'''
    exception = False
    try:
        req = requests.get(_api_url() + link, timeout=5)
    except requests.exceptions.RequestException:
        exception = True
    attempts = 1

    while exception or str(req.status_code).startswith('5'):
        if attempts > 5:
            raise DapiCommError('Could not connect to the API endpoint, sorry.')
        exception = False
        try:
            # Every second attempt, use the mirror
            req = requests.get(_api_url(attempts % 2) + link, timeout=5*attempts)
        except requests.exceptions.RequestException:
            exception = True
        attempts += 1

    return req
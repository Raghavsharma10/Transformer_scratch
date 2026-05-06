def _request(method, url, session=None, **kwargs):
    """Make HTTP request, raising an exception if it fails.
    """
    url = BASE_URL + url

    if session:
        request_func = getattr(session, method)
    else:
        request_func = getattr(requests, method)
    response = request_func(url, **kwargs)
    # raise an exception if request is not successful
    if not response.status_code == requests.codes.ok:
        raise DweepyError('HTTP {0} response'.format(response.status_code))
    response_json = response.json()
    if response_json['this'] == 'failed':
        raise DweepyError(response_json['because'])
    return response_json['with']
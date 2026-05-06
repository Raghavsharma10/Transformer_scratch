def listen_for_dweets_from(thing_name, timeout=900, key=None, session=None):
    """Create a real-time subscription to dweets
    """
    url = BASE_URL + '/listen/for/dweets/from/{0}'.format(thing_name)
    session = session or requests.Session()
    if key is not None:
        params = {'key': key}
    else:
        params = None

    start = datetime.datetime.utcnow()
    while True:
        request = requests.Request("GET", url, params=params).prepare()
        resp = session.send(request, stream=True, timeout=timeout)
        try:
            for x in _listen_for_dweets_from_response(resp):
                yield x
                _check_stream_timeout(start, timeout)
        except (ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            pass
        _check_stream_timeout(start, timeout)
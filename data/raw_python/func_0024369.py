def validate_request_timestamp(req_body, max_diff=150):
    """Ensure the request's timestamp doesn't fall outside of the
    app's specified tolerance.

    Returns True if this request is valid, False otherwise.

    :param req_body: JSON object parsed out of the raw POST data of a request.
    :param max_diff: Maximum allowable difference in seconds between request
        timestamp and system clock. Amazon requires <= 150 seconds for
        published skills.
    """

    time_str = req_body.get('request', {}).get('timestamp')

    if not time_str:
        log.error('timestamp not present %s', req_body)
        return False

    req_ts = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    diff = (datetime.utcnow() - req_ts).total_seconds()

    if abs(diff) > max_diff:
        log.error('timestamp difference too high: %d sec', diff)
        return False

    return True
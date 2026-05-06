def get_pause_duration(recursive_delay=5, default_duration=10):
    """
    Check the Overpass API status endpoint to determine how long to wait until
    next slot is available.

    Parameters
    ----------
    recursive_delay : int
        how long to wait between recursive calls if server is currently
        running a query
    default_duration : int
        if fatal error, function falls back on returning this value

    Returns
    -------
    pause_duration : int
    """
    try:
        response = requests.get('http://overpass-api.de/api/status')
        status = response.text.split('\n')[3]
        status_first_token = status.split(' ')[0]
    except Exception:
        # if status endpoint cannot be reached or output parsed, log error
        # and return default duration
        log('Unable to query http://overpass-api.de/api/status',
            level=lg.ERROR)
        return default_duration

    try:
        # if first token is numeric, it indicates the number of slots
        # available - no wait required
        available_slots = int(status_first_token)
        pause_duration = 0
    except Exception:
        # if first token is 'Slot', it tells you when your slot will be free
        if status_first_token == 'Slot':
            utc_time_str = status.split(' ')[3]
            utc_time = date_parser.parse(utc_time_str).replace(tzinfo=None)
            pause_duration = math.ceil(
                (utc_time - dt.datetime.utcnow()).total_seconds())
            pause_duration = max(pause_duration, 1)

        # if first token is 'Currently', it is currently running a query so
        # check back in recursive_delay seconds
        elif status_first_token == 'Currently':
            time.sleep(recursive_delay)
            pause_duration = get_pause_duration()

        else:
            # any other status is unrecognized - log an error and return
            # default duration
            log('Unrecognized server status: "{}"'.format(status),
                level=lg.ERROR)
            return default_duration

    return pause_duration
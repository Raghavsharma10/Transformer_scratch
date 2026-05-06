def overpass_request(data, pause_duration=None, timeout=180,
                     error_pause_duration=None):
    """
    Send a request to the Overpass API via HTTP POST and return the
    JSON response

    Parameters
    ----------
    data : dict or OrderedDict
        key-value pairs of parameters to post to Overpass API
    pause_duration : int
        how long to pause in seconds before requests, if None, will query
        Overpass API status endpoint
        to find when next slot is available
    timeout : int
        the timeout interval for the requests library
    error_pause_duration : int
        how long to pause in seconds before re-trying requests if error

    Returns
    -------
    response_json : dict
    """

    # define the Overpass API URL, then construct a GET-style URL
    url = 'http://www.overpass-api.de/api/interpreter'

    start_time = time.time()
    log('Posting to {} with timeout={}, "{}"'.format(url, timeout, data))
    response = requests.post(url, data=data, timeout=timeout)

    # get the response size and the domain, log result
    size_kb = len(response.content) / 1000.
    domain = re.findall(r'//(?s)(.*?)/', url)[0]
    log('Downloaded {:,.1f}KB from {} in {:,.2f} seconds'
        .format(size_kb, domain, time.time()-start_time))

    try:
        response_json = response.json()
        if 'remark' in response_json:
            log('Server remark: "{}"'.format(response_json['remark'],
                                             level=lg.WARNING))

    except Exception:
        # 429 = 'too many requests' and 504 = 'gateway timeout' from server
        # overload. handle these errors by recursively
        # calling overpass_request until a valid response is achieved
        if response.status_code in [429, 504]:
            # pause for error_pause_duration seconds before re-trying request
            if error_pause_duration is None:
                error_pause_duration = get_pause_duration()
            log('Server at {} returned status code {} and no JSON data. '
                'Re-trying request in {:.2f} seconds.'
                .format(domain, response.status_code, error_pause_duration),
                level=lg.WARNING)
            time.sleep(error_pause_duration)
            response_json = overpass_request(data=data,
                                             pause_duration=pause_duration,
                                             timeout=timeout)

        # else, this was an unhandled status_code, throw an exception
        else:
            log('Server at {} returned status code {} and no JSON data'
                .format(domain, response.status_code), level=lg.ERROR)
            raise Exception('Server returned no JSON data.\n{} {}\n{}'
                            .format(response, response.reason, response.text))

    return response_json
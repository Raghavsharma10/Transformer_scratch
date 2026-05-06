def discoverEndpoint(url, test_urls=True, headers={}, timeout=None, request=None, debug=False):
    """Discover any WebMention endpoint for a given URL.

    :param link: URL to discover WebMention endpoint
    :param test_urls: optional flag to test URLs for validation
    :param headers: optional headers to send with any web requests
    :type headers dict
    :param timeout: optional timeout for web requests
    :type timeout float
    :param request: optional Requests request object to avoid another GET
    :rtype: tuple (status_code, URL, [debug])
    """
    if test_urls:
        URLValidator(message='invalid URL')(url)

    # status, webmention
    endpointURL = None
    debugOutput = []
    try:
        if request is not None:
            targetRequest = request
        else:
            targetRequest = requests.get(url, verify=False, headers=headers, timeout=timeout)
        returnCode = targetRequest.status_code
        debugOutput.append('%s %s' % (returnCode, url))
        if returnCode == requests.codes.ok:
            try:
                linkHeader  = parse_link_header(targetRequest.headers['link'])
                endpointURL = linkHeader.get('webmention', '') or \
                              linkHeader.get('http://webmention.org', '') or \
                              linkHeader.get('http://webmention.org/', '') or \
                              linkHeader.get('https://webmention.org', '') or \
                              linkHeader.get('https://webmention.org/', '')
                # force searching in the HTML if not found
                if not endpointURL:
                    raise AttributeError
                debugOutput.append('found in link headers')
            except (KeyError, AttributeError):
                endpointURL = findEndpoint(targetRequest.text)
                debugOutput.append('found in body')
            if endpointURL is not None:
                endpointURL = urljoin(url, endpointURL)
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError, requests.exceptions.URLRequired,
            requests.exceptions.TooManyRedirects, requests.exceptions.Timeout):
        debugOutput.append('exception during GET request')
        returnCode = 500
    debugOutput.append('endpointURL: %s %s' % (returnCode, endpointURL))
    if debug:
        return (returnCode, endpointURL, debugOutput)
    else:
        return (returnCode, endpointURL)
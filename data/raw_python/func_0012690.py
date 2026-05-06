def sendWebmention(sourceURL, targetURL, webmention=None, test_urls=True, vouchDomain=None,
                   headers={}, timeout=None, debug=False):
    """Send to the :targetURL: a WebMention for the :sourceURL:

    The WebMention will be discovered if not given in the :webmention:
    parameter.

    :param sourceURL: URL that is referencing :targetURL:
    :param targetURL: URL of mentioned post
    :param webmention: optional WebMention endpoint
    :param test_urls: optional flag to test URLs for validation
    :param headers: optional headers to send with any web requests
    :type headers dict
    :param timeout: optional timeout for web requests
    :type timeout float

    :rtype: HTTPrequest object if WebMention endpoint was valid
    """
    if test_urls:
        v = URLValidator()
        v(sourceURL)
        v(targetURL)

    debugOutput = []
    originalURL = targetURL
    try:
        targetRequest = requests.get(targetURL)

        if targetRequest.status_code == requests.codes.ok:
            if len(targetRequest.history) > 0:
                redirect = targetRequest.history[-1]
                if (redirect.status_code == 301 or redirect.status_code == 302) and 'Location' in redirect.headers:
                    targetURL = urljoin(targetURL, redirect.headers['Location'])
                    debugOutput.append('targetURL redirected: %s' % targetURL)
        if webmention is None:
            wStatus, wUrl = discoverEndpoint(targetURL, headers=headers, timeout=timeout, request=targetRequest)
        else:
            wStatus = 200
            wUrl = webmention
        debugOutput.append('endpointURL: %s %s' % (wStatus, wUrl))
        if wStatus == requests.codes.ok and wUrl is not None:
            if test_urls:
                v(wUrl)
            payload = {'source': sourceURL,
                       'target': originalURL}
            if vouchDomain is not None:
                payload['vouch'] = vouchDomain
            try:
                result = requests.post(wUrl, data=payload, headers=headers, timeout=timeout)
                debugOutput.append('POST %s -- %s' % (wUrl, result.status_code))
                if result.status_code == 405 and len(result.history) > 0:
                    redirect = result.history[-1]
                    if redirect.status_code == 301 and 'Location' in redirect.headers:
                        result = requests.post(redirect.headers['Location'], data=payload, headers=headers, timeout=timeout)
                        debugOutput.append('redirected POST %s -- %s' % (redirect.headers['Location'], result.status_code))
            except Exception as e:
                result = None
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError, requests.exceptions.URLRequired,
            requests.exceptions.TooManyRedirects, requests.exceptions.Timeout):
        debugOutput.append('exception during GET request')
        result = None
    return result
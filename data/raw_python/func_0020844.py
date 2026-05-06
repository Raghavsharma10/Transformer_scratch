def sailthru_http_request(url, data, method, file_data=None, headers=None, request_timeout=10):
    """
    Perform an HTTP GET / POST / DELETE request
    """
    data = flatten_nested_hash(data)
    method = method.upper()
    params, data = (None, data) if method == 'POST' else (data, None)
    sailthru_headers = {'User-Agent': 'Sailthru API Python Client %s; Python Version: %s' % ('2.3.5', platform.python_version())}
    if headers and isinstance(headers, dict):
        for key, value in sailthru_headers.items():
            headers[key] = value
    else:
        headers = sailthru_headers
    try:
        response = requests.request(method, url, params=params, data=data, files=file_data, headers=headers, timeout=request_timeout)
        return SailthruResponse(response)
    except requests.HTTPError as e:
        raise SailthruClientError(str(e))
    except requests.RequestException as e:
        raise SailthruClientError(str(e))
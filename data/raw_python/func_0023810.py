def get_wsgi_request_object(curr_request, method, url, headers, body):
    '''
        Based on the given request parameters, constructs and returns the WSGI request object.
    '''
    x_headers = headers_to_include_from_request(curr_request)
    method, t_headers = pre_process_method_headers(method, headers)

    # Add default content type.
    if "CONTENT_TYPE" not in t_headers:
        t_headers.update({"CONTENT_TYPE": _settings.DEFAULT_CONTENT_TYPE})

    # Override existing batch requests headers with the new headers passed for this request.
    x_headers.update(t_headers)

    content_type = x_headers.get("CONTENT_TYPE", _settings.DEFAULT_CONTENT_TYPE)

    # Get hold of request factory to construct the request.
    _request_factory = BatchRequestFactory()
    _request_provider = getattr(_request_factory, method)

    secure = _settings.USE_HTTPS

    request = _request_provider(url, data=body, secure=secure,
                                content_type=content_type, **x_headers)

    return request
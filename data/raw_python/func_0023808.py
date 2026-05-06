def pre_process_method_headers(method, headers):
    '''
        Returns the lowered method.
        Capitalize headers, prepend HTTP_ and change - to _.
    '''
    method = method.lower()

    # Standard WSGI supported headers
    _wsgi_headers = ["content_length", "content_type", "query_string",
                     "remote_addr", "remote_host", "remote_user",
                     "request_method", "server_name", "server_port"]

    _transformed_headers = {}

    # For every header, replace - to _, prepend http_ if necessary and convert
    # to upper case.
    for header, value in headers.items():

        header = header.replace("-", "_")
        header = "http_{header}".format(
            header=header) if header.lower() not in _wsgi_headers else header
        _transformed_headers.update({header.upper(): value})

    return method, _transformed_headers
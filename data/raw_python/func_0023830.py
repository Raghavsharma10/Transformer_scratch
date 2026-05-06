def get_wsgi_requests(request):
    '''
        For the given batch request, extract the individual requests and create
        WSGIRequest object for each.
    '''
    valid_http_methods = ["get", "post", "put", "patch", "delete", "head", "options", "connect", "trace"]
    requests = json.loads(request.body)

    if type(requests) not in (list, tuple):
        raise BadBatchRequest("The body of batch request should always be list!")

    # Max limit check.
    no_requests = len(requests)

    if no_requests > _settings.MAX_LIMIT:
        raise BadBatchRequest("You can batch maximum of %d requests." % (_settings.MAX_LIMIT))

    # We could mutate the current request with the respective parameters, but mutation is ghost in the dark,
    # so lets avoid. Construct the new WSGI request object for each request.

    def construct_wsgi_from_data(data):
        '''
            Given the data in the format of url, method, body and headers, construct a new
            WSGIRequest object.
        '''
        url = data.get("url", None)
        method = data.get("method", None)

        if url is None or method is None:
            raise BadBatchRequest("Request definition should have url, method defined.")

        if method.lower() not in valid_http_methods:
            raise BadBatchRequest("Invalid request method.")

        body = data.get("body", "")
        headers = data.get("headers", {})
        return get_wsgi_request_object(request, method, url, headers, body)

    return [construct_wsgi_from_data(data) for data in requests]
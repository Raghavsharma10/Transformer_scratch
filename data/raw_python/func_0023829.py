def get_response(wsgi_request):
    '''
        Given a WSGI request, makes a call to a corresponding view
        function and returns the response.
    '''
    service_start_time = datetime.now()
    # Get the view / handler for this request
    view, args, kwargs = resolve(wsgi_request.path_info)

    kwargs.update({"request": wsgi_request})

    # Let the view do his task.
    try:
        resp = view(*args, **kwargs)
    except Exception as exc:
        resp = HttpResponseServerError(content=exc.message)

    headers = dict(resp._headers.values())
    # Convert HTTP response into simple dict type.
    d_resp = {"status_code": resp.status_code, "reason_phrase": resp.reason_phrase,
              "headers": headers}
    try:
        d_resp.update({"body": resp.content})
    except ContentNotRenderedError:
        resp.render()
        d_resp.update({"body": resp.content})

    # Check if we need to send across the duration header.
    if _settings.ADD_DURATION_HEADER:
        d_resp['headers'].update({_settings.DURATION_HEADER_NAME: (datetime.now() - service_start_time).seconds})

    return d_resp
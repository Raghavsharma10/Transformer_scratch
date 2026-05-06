def handle_batch_requests(request, *args, **kwargs):
    '''
        A view function to handle the overall processing of batch requests.
    '''
    batch_start_time = datetime.now()
    try:
        # Get the Individual WSGI requests.
        wsgi_requests = get_wsgi_requests(request)
    except BadBatchRequest as brx:
        return HttpResponseBadRequest(content=brx.message)

    # Fire these WSGI requests, and collect the response for the same.
    response = execute_requests(wsgi_requests)

    # Evrything's done, return the response.
    resp = HttpResponse(
        content=json.dumps(response), content_type="application/json")

    if _settings.ADD_DURATION_HEADER:
        resp.__setitem__(_settings.DURATION_HEADER_NAME, str((datetime.now() - batch_start_time).seconds))
    return resp
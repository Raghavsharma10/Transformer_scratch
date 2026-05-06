def handleHttpPost(request, endpoint):
    """
    Handles the specified HTTP POST request, which maps to the specified
    protocol handler endpoint and protocol request class.
    """
    if request.mimetype and request.mimetype != MIMETYPE:
        raise exceptions.UnsupportedMediaTypeException()
    request = request.get_data()
    if request == '' or request is None:
        request = '{}'
    responseStr = endpoint(request)
    return getFlaskResponse(responseStr)
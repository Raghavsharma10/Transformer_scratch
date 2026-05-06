def jsonp_response(data, callback="f", status=200, serializer=None):
    """
    Returns an HttpResponse object containing JSON serialized data,
    wrapped in a JSONP callback.

    The mime-type is set to application/x-javascript, and the charset to UTF-8.
    """
    val = json.dumps(data, default=serializer)
    ret = "{callback}('{val}');".format(callback=callback, val=val)

    return HttpResponse(ret,
                        status=status,
                        content_type='application/x-javascript; charset=UTF-8')
def eventsource_connect(url, io_loop=None, callback=None, connect_timeout=None):
    """Client-side eventsource support.

    Takes a url and returns a Future whose result is a
    `EventSourceClient`.

    """
    if io_loop is None:
        io_loop = IOLoop.current()
    if isinstance(url, httpclient.HTTPRequest):
        assert connect_timeout is None
        request = url
        # Copy and convert the headers dict/object (see comments in
        # AsyncHTTPClient.fetch)
        request.headers = httputil.HTTPHeaders(request.headers)
    else:
        request = httpclient.HTTPRequest(
            url,
            connect_timeout=connect_timeout,
            headers=httputil.HTTPHeaders({
                "Accept-Encoding": "identity"
            })
        )
    request = httpclient._RequestProxy(
        request, httpclient.HTTPRequest._DEFAULTS)
    conn = EventSourceClient(io_loop, request)
    if callback is not None:
        io_loop.add_future(conn.connect_future, callback)
    return conn.connect_future
async def rewrite_middleware(server, request):
    '''
    Sanic middleware that utilizes a security class's "rewrite" method to
    check
    '''
    if singletons.settings.SECURITY is not None:
        security_class = singletons.settings.load('SECURITY')
    else:
        security_class = DummySecurity
    security = security_class()
    try:
        new_path = await security.rewrite(request)
    except SecurityException as e:
        msg = ''
        if DEBUG:
            msg = str(e)
        return server.response.text(msg, status=400)
    request.path = new_path
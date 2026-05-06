def make_middleware(app=None, *args, **kw):
    """ Given an app, return that app wrapped in RaptorizeMiddleware """
    app = RaptorizeMiddleware(app, *args, **kw)
    return app
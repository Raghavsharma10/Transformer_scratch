def app_1(env, start_response):
    """This is simple WSGI application that will be served by uWSGI."""

    from uwsgiconf.runtime.environ import uwsgi_env

    start_response('200 OK', [('Content-Type','text/html')])

    data = [
        '<h1>uwsgiconf demo: one file</h1>',

        '<div>uWSGI version: %s</div>' % uwsgi_env.get_version(),
        '<div>uWSGI request ID: %s</div>' % uwsgi_env.request.id,
    ]

    return encode(data)
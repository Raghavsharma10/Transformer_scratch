def app_2(env, start_response):
    """This is another simple WSGI application that will be served by uWSGI."""

    import random

    start_response('200 OK', [('Content-Type','text/html')])

    data = [
        '<h1>uwsgiconf demo: one file second app</h1>',

        '<div>Some random number for you: %s</div>' % random.randint(1, 99999),
    ]

    return encode(data)
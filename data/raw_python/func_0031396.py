def connect(uri):
    """
    Connects to an nREPL endpoint identified by the given URL/URI.  Valid
    examples include:

      nrepl://192.168.0.12:7889
      telnet://localhost:5000
      http://your-app-name.heroku.com/repl

    This fn delegates to another looked up in  that dispatches on the scheme of
    the URI provided (which can be a string or java.net.URI).  By default, only
    `nrepl` (corresponding to using the default bencode transport) is
    supported. Alternative implementations may add support for other schemes,
    such as http/https, JMX, various message queues, etc.
    """
    #
    uri = uri if isinstance(uri, ParseResult) else urlparse(uri)
    if not uri.scheme:
        raise ValueError("uri has no scheme: " + uri)
    f = _connect_fns.get(uri.scheme.lower(), None)
    if not f:
        err = "No connect function registered for scheme `%s`" % uri.scheme
        raise Exception(err)
    return f(uri)
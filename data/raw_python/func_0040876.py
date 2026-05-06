def background(cl, proto=EchoProcess, **kw):
    """
    Use the reactor to run a process in the background.

    Keep the pid around.

    ``proto'' may be any callable which returns an instance of ProcessProtocol
    """
    if isinstance(cl, basestring):
        cl = shlex.split(cl)

    if not cl[0].startswith('/'):
        path = which(cl[0])
        assert path, '%s not found' % cl[0]
        cl[0] = path[0]

    d = Deferred()
    proc = reactor.spawnProcess(
            proto(name=basename(cl[0]), deferred=d),
            cl[0],
            cl,
            env=os.environ,
            **kw)

    daycare.add(proc.pid)
    return d
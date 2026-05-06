def doublefork(pidfile, logfile, cwd, umask): # pragma: nocover
    '''Daemonize current process.
    After first fork we return to the shell and removing our self from
    controling terminal via `setsid`.
    After second fork we are not session leader any more and cant get
    controlling terminal when opening files.'''
    try:
        if os.fork():
            os._exit(os.EX_OK)
    except OSError as e:
        sys.exit('fork #1 failed: ({}) {}'.format(e.errno, e.strerror))
    os.setsid()
    os.chdir(cwd)
    os.umask(umask)
    try:
        if os.fork():
            os._exit(os.EX_OK)
    except OSError as e:
        sys.exit('fork #2 failed: ({}) {}'.format(e.errno, e.strerror))
    if logfile is not None:
        si = open('/dev/null')
        if six.PY2:
            so = open(logfile, 'a+', 0)
        else:
            so = io.open(logfile, 'ab+', 0)
            so = io.TextIOWrapper(so, write_through=True, encoding="utf-8")

        os.dup2(si.fileno(), 0)
        os.dup2(so.fileno(), 1)
        os.dup2(so.fileno(), 2)
        sys.stdin = si
        sys.stdout = sys.stderr = so
    with open(pidfile, 'w') as f:
        f.write(str(os.getpid()))
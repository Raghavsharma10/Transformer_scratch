def write_pidfile(pidfile):
    """write current pid to pidfile.
    """
    pid = os.getpid()
    if pidfile:
        with open(pidfile, "w", encoding="utf-8") as fobj:
            fobj.write(six.u(str(pid)))
    return pid
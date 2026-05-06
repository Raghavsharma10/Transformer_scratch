def load_pid(pidfile):
    """read pid from pidfile.
    """
    if pidfile and os.path.isfile(pidfile):
        with open(pidfile, "r", encoding="utf-8") as fobj:
            return int(fobj.readline().strip())
    return 0
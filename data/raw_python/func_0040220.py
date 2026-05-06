def daemon_stop(pidfile, sig=None):
    """Stop application.
    """
    logger.debug("stop daemon application pidfile={pidfile}.".format(pidfile=pidfile))
    pid = load_pid(pidfile)
    logger.debug("load pid={pid}".format(pid=pid))
    if not pid:
        six.print_("Application is not running or crashed...", file=os.sys.stderr)
        os.sys.exit(195)
    process_kill(pid, sig)
    return pid
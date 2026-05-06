def clean_pid_file(pidfile):
    """clean pid file.
    """
    if pidfile and os.path.exists(pidfile):
        os.unlink(pidfile)
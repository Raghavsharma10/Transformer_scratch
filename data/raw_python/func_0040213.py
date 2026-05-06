def make_basic_daemon(workspace=None):
    """Make basic daemon.
    """
    workspace = workspace or os.getcwd()
    # first fork
    if os.fork():
        os._exit(0)
    # change env
    os.chdir(workspace)
    os.setsid()
    os.umask(0o22)
    # second fork
    if os.fork():
        os._exit(0)
    # reset stdin/stdout/stderr to /dev/null
    null = os.open('/dev/null', os.O_RDWR)
    try:
        for i in range(0, 3):
            try:
                os.dup2(null, i)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
    finally:
        os.close(null)
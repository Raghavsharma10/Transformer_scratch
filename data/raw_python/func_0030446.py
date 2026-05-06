def count_open_fds():
    """return the number of open file descriptors for current process.

    .. warning: will only work on UNIX-like os-es.

    http://stackoverflow.com/a/7142094

    """

    pid = os.getpid()
    procs = subprocess.check_output(
        ['lsof', '-w', '-Ff', '-p', str(pid)])

    nprocs = len(
        [s for s in procs.split('\n') if s and s[0] == 'f' and s[1:].isdigit()]
    )
    return nprocs
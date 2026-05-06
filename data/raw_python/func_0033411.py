def daemonize(pid_file=None, cwd=None):
    """
    Detach a process from the controlling terminal and run it in the
    background as a daemon.

    Modified version of:
        code.activestate.com/recipes/278731-creating-a-daemon-the-python-way/

    author = "Chad J. Schroeder"
    copyright = "Copyright (C) 2005 Chad J. Schroeder"
    """
    cwd = cwd or '/'
    try:
        pid = os.fork()
    except OSError as e:
        raise Exception("%s [%d]" % (e.strerror, e.errno))

    if (pid == 0):   # The first child.
        os.setsid()
        try:
            pid = os.fork()    # Fork a second child.
        except OSError as e:
            raise Exception("%s [%d]" % (e.strerror, e.errno))
        if (pid == 0):    # The second child.
            os.chdir(cwd)
            os.umask(0)
        else:
            os._exit(0)    # Exit parent (the first child) of the second child.
    else:
        os._exit(0)   # Exit parent of the first child.

    maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if (maxfd == resource.RLIM_INFINITY):
        maxfd = 1024

    # Iterate through and close all file descriptors.
    for fd in range(0, maxfd):
        try:
            os.close(fd)
        except OSError:   # ERROR, fd wasn't open to begin with (ignored)
            pass

    os.open('/dev/null', os.O_RDWR)  # standard input (0)

    # Duplicate standard input to standard output and standard error.
    os.dup2(0, 1)            # standard output (1)
    os.dup2(0, 2)            # standard error (2)

    pid_file = pid_file or '%s.pid' % os.getpid()
    write_file(pid_file, os.getpid())
    return 0
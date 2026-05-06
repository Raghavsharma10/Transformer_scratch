def safe_popen(*args, **kwargs):
    '''This wrapper works around two major deadlock issues to do with pipes.
    The first is that, before Python 3.2 on POSIX systems, os.pipe() creates
    inheritable file descriptors, which leak to all child processes and prevent
    reads from reaching EOF. The workaround for this is to set close_fds=True
    on POSIX, which was not the default in those versions. See PEP 0446 for
    many details.

    The second issue arises on Windows, where we're not allowed to set
    close_fds=True while also setting stdin/stdout/stderr. Descriptors from
    os.pipe() on Windows have never been inheritable, so it would seem that
    we're safe. However, the Windows implementation of subprocess.Popen()
    creates temporary inheritable copies of its descriptors, and these can
    leak. The workaround for this is to protect Popen() with a global lock. See
    https://bugs.python.org/issue25565.'''

    close_fds = (os.name != 'nt')
    with popen_lock:
        return subprocess.Popen(*args, close_fds=close_fds, **kwargs)
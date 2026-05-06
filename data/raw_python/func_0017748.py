def _detach_process():
    """
    Detach daemon process.

    Forks the current process into a parent and a detached child. The
    child process resides in its own process group, has no controlling
    terminal attached and is cleaned up by the init process.

    Returns ``True`` for the parent and ``False`` for the child.
    """
    # To detach from our process group we need to call ``setsid``. We
    # can only do that if we aren't a process group leader. Therefore
    # we fork once, which makes sure that the new child process is not
    # a process group leader.
    pid = os.fork()
    if pid > 0:
        # Parent process
        # Use waitpid to "collect" the child process and avoid Zombies
        os.waitpid(pid, 0)
        return True
    os.setsid()
    # We now fork a second time and let the second's fork parent exit.
    # This makes the second fork's child process an orphan. Orphans are
    # cleaned up by the init process, so we won't end up with a zombie.
    # In addition, the second fork's child is no longer a session
    # leader and can therefore never acquire a controlling terminal.
    pid = os.fork()
    if pid > 0:
        os._exit(os.EX_OK)
    return False
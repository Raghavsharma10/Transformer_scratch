def os_forkpty_wrapper():
    """Wraps os.forkpty() to run atfork handlers."""
    pid = None
    prepare_to_fork_acquire()
    try:
        pid, fd = _orig_os_forkpty()
    finally:
        if pid == 0:
            child_after_fork_release()
        else:
            parent_after_fork_release()
    return pid, fd
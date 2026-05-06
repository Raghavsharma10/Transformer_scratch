def monkeypatch_os_fork_functions():
    """
    Replace os.fork* with wrappers that use ForkSafeLock to acquire
    all locks before forking and release them afterwards.
    """
    builtin_function = type(''.join)
    if hasattr(os, 'fork') and isinstance(os.fork, builtin_function):
        global _orig_os_fork
        _orig_os_fork = os.fork
        os.fork = os_fork_wrapper
    if hasattr(os, 'forkpty') and isinstance(os.forkpty, builtin_function):
        global _orig_os_forkpty
        _orig_os_forkpty = os.forkpty
        os.forkpty = os_forkpty_wrapper
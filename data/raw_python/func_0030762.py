def terminate(pid, sig, timeout):
    '''Terminates process with PID `pid` and returns True if process finished
    during `timeout`. Current user must have permission to access process
    information.'''
    os.kill(pid, sig)
    start = time.time()
    while True:
        try:
            # This is requireed if it's our child to avoid zombie. Also
            # is_running() returns True for zombie process.
            _, status = os.waitpid(pid, os.WNOHANG)
        except OSError as exc:
            if exc.errno != errno.ECHILD: # pragma: nocover
                raise
        else:
            if status:
                return True
        if not is_running(pid):
            return True
        if time.time()-start>=timeout:
            return False
        time.sleep(0.1)
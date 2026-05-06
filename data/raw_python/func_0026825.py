def kill(pid_file):
    """
    Attempts to shut down a previously started daemon.
    """
    try:
        with open(pid_file) as f:
            os.kill(int(f.read()), 9)
        os.remove(pid_file)
    except (IOError, OSError):
        return False
    return True
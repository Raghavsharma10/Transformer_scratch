def _runshell(cmd, exception):
    """ Run a shell command. if fails, raise a proper exception. """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.wait() != 0:
        raise BridgeException(exception)
    return p
def is_installed(prog):
    """Return whether or not a given executable is installed on the machine."""
    with open(os.devnull, 'w') as devnull:
        try:
            if os.name == 'nt':
                retcode = subprocess.call(['where', prog], stdout=devnull)
            else:
                retcode = subprocess.call(['which', prog], stdout=devnull)
        except OSError as e:
            # If where or which doesn't exist, a "ENOENT" error will occur (The
            # FileNotFoundError subclass on Python 3).
            if e.errno != errno.ENOENT:
                raise
            retcode = 1

    return retcode == 0
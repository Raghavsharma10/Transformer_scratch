def path_join(*args):
    """
    Wrapper around `os.path.join`.
    Makes sure to join paths of the same type (bytes).
    """
    args = (paramiko.py3compat.u(arg) for arg in args)
    return os.path.join(*args)
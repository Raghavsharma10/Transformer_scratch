def which(executable_name, env_var='PATH'):
    """Equivalent to ``which executable_name`` in a *nix environment.

    Will return ``None`` if ``executable_name`` cannot be found in ``env_var``
    or if ``env_var`` is not set. Otherwise will return the first match in
    ``env_var``.

    Note: this function will likely not work on Windows.

    Code taken and modified from:
        http://www.velocityreviews.com/forums/
        t689526-python-library-call-equivalent-to-which-command.html

    """
    exec_fp = None

    if env_var in os.environ:
        paths = os.environ[env_var]

        for path in paths.split(os.pathsep):
            curr_exec_fp = os.path.join(path, executable_name)

            if os.access(curr_exec_fp, os.X_OK):
                exec_fp = curr_exec_fp
                break

    return exec_fp
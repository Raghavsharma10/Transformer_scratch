def which(program, environ=None):
    """
    Find out if an executable exists in the supplied PATH.
    If so, the absolute path to the executable is returned.
    If not, an exception is raised.

    :type string
    :param program: Executable to be checked for

    :param dict
    :param environ: Any additional ENV variables required, specifically PATH

    :return string|:class:`command.CommandException` Returns the location if found, otherwise raises exception
    """
    def is_exe(path):
        """
        Helper method to check if a file exists and is executable
        """
        return isfile(path) and os.access(path, os.X_OK)

    if program is None:
        raise CommandException("Invalid program name passed")

    fpath, fname = split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        if environ is None:
            environ = os.environ

        for path in environ['PATH'].split(os.pathsep):
            exe_file = join(path, program)
            if is_exe(exe_file):
                return exe_file

    raise CommandException("Could not find %s" % program)
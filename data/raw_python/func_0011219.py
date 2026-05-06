def parent():
    """Determine subshell matching the currently running shell

    The shell is determined by either a pre-defined BE_SHELL
    environment variable, or, if none is found, via psutil
    which looks at the parent process directly through
    system-level calls.

    For example, is `be` is run from cmd.exe, then the full
    path to cmd.exe is returned, and the same goes for bash.exe
    and bash (without suffix) for Unix environments.

    The point is to return an appropriate subshell for the
    running shell, as opposed to the currently running OS.

    """

    if self._parent:
        return self._parent

    if "BE_SHELL" in os.environ:
        self._parent = os.environ["BE_SHELL"]
    else:
        # If a shell is not provided, rely on `psutil`
        # to look at the calling process name.
        try:
            import psutil
        except ImportError:
            raise ImportError(
                "No shell provided, see documentation for "
                "BE_SHELL for more information.\n"
                "https://github.com/mottosso/be/wiki"
                "/environment#read-environment-variables")

        parent = psutil.Process(os.getpid()).parent()

        # `pip install` creates an additional executable
        # that tricks the above mechanism to think of it
        # as the parent shell. See #34 for more.
        if parent.name() in ("be", "be.exe"):
            parent = parent.parent()

        self._parent = str(parent.exe())

    return self._parent
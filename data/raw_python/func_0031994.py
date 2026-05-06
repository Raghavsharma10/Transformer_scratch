def init_run(shell, no_daemon, daemon_options, daemon_outfile):
    """
    Configure your shell.

    Add the following line in your shell RC file and then you are
    ready to go::

      eval $(%(prog)s)

    To check if your shell is supported, simply run::

      %(prog)s --no-daemon

    If you want to specify shell other than $SHELL, you can give
    --shell option::

      eval $(%(prog)s --shell zsh)

    By default, this command also starts daemon in background to
    automatically index shell history records.  To not start daemon,
    use --no-daemon option like this::

      eval $(%(prog)s --no-daemon)

    To see the other methods to launch the daemon process, see
    ``rash daemon --help``.

    """
    import sys
    from .__init__ import __version__
    init_file = find_init(shell)
    if os.path.exists(init_file):
        sys.stdout.write(INIT_TEMPLATE.format(
            file=init_file, version=__version__))
    else:
        raise RuntimeError(
            "Shell '{0}' is not supported.".format(shell_name(shell)))

    if not no_daemon:
        from .daemon import start_daemon_in_subprocess
        start_daemon_in_subprocess(daemon_options, daemon_outfile)
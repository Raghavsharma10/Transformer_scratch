def start(client, container, interactive=True, stdout=None, stderr=None, stdin=None, **kwargs):
    """
    Present the PTY of the container inside the current process.

    This is just a wrapper for PseudoTerminal(client, container).start()
    """

    PseudoTerminal(client, container, interactive=interactive, stdout=stdout, stderr=stderr, stdin=stdin).start(**kwargs)
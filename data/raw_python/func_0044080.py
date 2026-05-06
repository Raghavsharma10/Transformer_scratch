def terminal_reserve(progress_obj, terminal_obj=None, identifier=None):
    """ Registers the terminal (stdout) for printing.

    Useful to prevent multiple processes from writing progress bars
    to stdout.

    One process (server) prints to stdout and a couple of subprocesses
    do not print to the same stdout, because the server has reserved it.
    Of course, the clients have to be nice and check with
    terminal_reserve first if they should (not) print.
    Nothing is locked.

    Returns
    -------
    True if reservation was successful (or if we have already reserved this tty),
    False if there already is a reservation from another instance.
    """
    if terminal_obj is None:
        terminal_obj = sys.stdout

    if identifier is None:
        identifier = ''

    if terminal_obj in TERMINAL_RESERVATION:  # terminal was already registered
        log.debug("this terminal %s has already been added to reservation list", terminal_obj)

        if TERMINAL_RESERVATION[terminal_obj] is progress_obj:
            log.debug("we %s have already reserved this terminal %s", progress_obj, terminal_obj)
            return True
        else:
            log.debug("someone else %s has already reserved this terminal %s", TERMINAL_RESERVATION[terminal_obj],
                      terminal_obj)
            return False
    else:  # terminal not yet registered
        log.debug("terminal %s was reserved for us %s", terminal_obj, progress_obj)
        TERMINAL_RESERVATION[terminal_obj] = progress_obj
        return True
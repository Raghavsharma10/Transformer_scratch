def terminal_unreserve(progress_obj, terminal_obj=None, verbose=0, identifier=None):
    """ Unregisters the terminal (stdout) for printing.

    an instance (progress_obj) can only unreserve the tty (terminal_obj) when it also reserved it

    see terminal_reserved for more information

    Returns
    -------
    None
    """

    if terminal_obj is None:
        terminal_obj = sys.stdout

    if identifier is None:
        identifier = ''
    else:
        identifier = identifier + ': '

    po = TERMINAL_RESERVATION.get(terminal_obj)
    if po is None:
        log.debug("terminal %s was not reserved, nothing happens", terminal_obj)
    else:
        if po is progress_obj:
            log.debug("terminal %s now unreserned", terminal_obj)
            del TERMINAL_RESERVATION[terminal_obj]
        else:
            log.debug("you %s can NOT unreserve terminal %s be cause it was reserved by %s", progress_obj, terminal_obj,
                      po)
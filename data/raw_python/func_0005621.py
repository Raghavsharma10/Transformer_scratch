def devserver(port, admin_port, clear):
    # type: (int, int, bool) -> None
    """ Run devserver. """
    from . import logic

    logic.devserver(port, admin_port, clear)
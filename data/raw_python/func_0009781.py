def init_controller(url):
    """Initialize a controller.

    Provides a single global controller for applications that can't do this
    themselves
    """
    # pylint: disable=global-statement
    global _VERA_CONTROLLER
    created = False
    if _VERA_CONTROLLER is None:
        _VERA_CONTROLLER = VeraController(url)
        created = True
        _VERA_CONTROLLER.start()
    return [_VERA_CONTROLLER, created]
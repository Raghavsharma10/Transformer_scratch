def fatal(msg, exitcode=1, **kwargs):
    """Prints a message then exits the program. Optionally pause before exit
    with `pause=True` kwarg."""
    # NOTE: Can't use normal arg named `pause` since function has same name.
    pause_before_exit = kwargs.pop("pause") if "pause" in kwargs.keys() else False
    echo("[FATAL] " + msg, **kwargs)
    if pause_before_exit:
        pause()
    sys.exit(exitcode)
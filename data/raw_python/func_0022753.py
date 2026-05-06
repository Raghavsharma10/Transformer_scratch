def get_dpi(raise_error=True):
    """Get screen DPI from the OS

    Parameters
    ----------
    raise_error : bool
        If True, raise an error if DPI could not be determined.

    Returns
    -------
    dpi : float
        Dots per inch of the primary screen.
    """
    display = quartz.CGMainDisplayID()
    mm = quartz.CGDisplayScreenSize(display)
    px = quartz.CGDisplayBounds(display).size
    return (px.width/mm.width + px.height/mm.height) * 0.5 * 25.4
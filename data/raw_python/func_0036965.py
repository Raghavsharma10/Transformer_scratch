def color_for_level(level):
    """
    Returns the colorama Fore color for a given log level.

    If color is not available, returns None.
    """
    if not color_available:
        return None

    return {
        logging.DEBUG: colorama.Fore.WHITE,
        logging.INFO: colorama.Fore.BLUE,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.CRITICAL: colorama.Fore.MAGENTA
    }.get(level, colorama.Fore.WHITE)
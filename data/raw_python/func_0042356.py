def get_printer(colors: bool = True, width_limit: bool = True, disabled: bool = False) -> Printer:
    """
    Returns an already initialized instance of the printer.

    :param colors: If False, no colors will be printed.
    :param width_limit: If True, printing width will be limited by console width.
    :param disabled: If True, nothing will be printed.
    """
    global _printer
    global _colors
    # Make sure we can print colors if needed.
    colors = colors and _colors
    # If the printer was never defined before, or the settings have changed.
    if not _printer or (colors != _printer._colors) or (width_limit != _printer._width_limit):
        _printer = Printer(DefaultWriter(disabled=disabled), colors=colors, width_limit=width_limit)
    return _printer
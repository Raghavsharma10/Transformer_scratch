def get_console_width() -> int:
    """
    A small utility function for getting the current console window's width.

    :return: The current console window's width.
    """
    # Assigning the value once, as frequent call to this function
    # causes a major slow down(ImportErrors + isinstance).
    global _IN_QT
    if _IN_QT is None:
        _IN_QT = _in_qtconsole()

    try:
        if _IN_QT:
            # QTConsole determines and handles the max line length by itself.
            width = sys.maxsize
        else:
            width = _get_windows_console_width() if os.name == 'nt' else _get_linux_console_width()
        if width <= 0:
            return 80
        return width
    except Exception:
        # Default value.
        return 80
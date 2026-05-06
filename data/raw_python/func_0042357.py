def _get_windows_console_width() -> int:
    """
    A small utility function for getting the current console window's width, in Windows.

    :return: The current console window's width.
    """
    from ctypes import byref, windll
    import pyreadline

    out = windll.kernel32.GetStdHandle(-11)
    info = pyreadline.console.CONSOLE_SCREEN_BUFFER_INFO()
    windll.kernel32.GetConsoleScreenBufferInfo(out, byref(info))
    return info.dwSize.X
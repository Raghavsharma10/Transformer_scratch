def title(msg):
    """Sets the title of the console window."""
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(tounicode(msg))
def clear():
    """Clears the console."""
    if sys.platform.startswith("win"):
        call("cls", shell=True)
    else:
        call("clear", shell=True)
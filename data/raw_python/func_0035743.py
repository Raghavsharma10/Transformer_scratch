def getuser():
    """Return the username."""
    is_windows = platform.system() == "Windows"
    no_username_in_env = os.environ.get("USERNAME") is None
    return cross_platform_getuser(is_windows, no_username_in_env)
def print_debug(*args, **kwargs):
    """
    Print if and only if the debug flag is set true in the config.yaml file.

    Args:
        args : var args of print arguments.

    """
    if WTF_CONFIG_READER.get("debug", False) == True:
        print(*args, **kwargs)
def file_root_name(name):
    """
    Returns the root name of a file from a full file path.

    It will not raise an error if the result is empty, but an warning will be
    issued.
    """
    base = os.path.basename(name)
    root = os.path.splitext(base)[0]
    if not root:
        warning = 'file_root_name returned an empty root name from \"{0}\"'
        log.warning(warning.format(name))
    return root
def add_pythonpath(path):
    """
    Prepend given path to environment variable PYTHONPATH.

    :param path: Path to add to PYTHONPATH.

    :return: New PYTHONPATH value.
    """
    # Get PYTHONPATH value. Default is empty string.
    pythonpath = os.environ.setdefault('PYTHONPATH', '')

    # If given path is not in PYTHONPATH
    if path not in pythonpath.split(os.pathsep):
        # Prepend given path to PYTHONPATH
        pythonpath = os.environ['PYTHONPATH'] = \
            (path + os.pathsep + pythonpath) if pythonpath else path

    # Return new PYTHONPATH value
    return pythonpath
def load(filepath=None):
    """
        Reads a .env file into os.environ.

        For a set filepath, open the file and read contents into os.environ.
        If filepath is not set then look in current dir for a .env file.
    """
    if filepath and os.path.exists(filepath):
        pass
    else:
        if not os.path.exists('.env'):
            return False
        filepath = os.path.join('.env')

    for key, value in _get_line_(filepath):
        # set the key, value in the python environment vars dictionary
        # does not make modifications system wide.
        os.environ.setdefault(key, str(value))
    return True
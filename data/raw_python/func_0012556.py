def pushd(directory):
    """Change working directories in style and stay organized!

    :param directory: Where do you want to go and remember?
    :return: saved directory stack
    """
    directory = os.path.expanduser(directory)
    _saved_paths.insert(0, os.path.abspath(os.getcwd()))
    os.chdir(directory)
    return [directory] + _saved_paths
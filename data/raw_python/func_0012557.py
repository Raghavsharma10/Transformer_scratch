def popd():
    """Go back to where you once were.

    :return: saved directory stack
    """
    try:
        directory = _saved_paths.pop(0)
    except IndexError:
        return [os.getcwd()]
    os.chdir(directory)
    return [directory] + _saved_paths
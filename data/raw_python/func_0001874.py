def load_file_to_str(path):
    # type: (str) -> str
    """
    Load file into a string removing newlines

    Args:
        path (str): Path to file

    Returns:
        str: String contents of file

    """
    with open(path, 'rt') as f:
        string = f.read().replace(linesep, '')
    if not string:
        raise LoadError('%s file is empty!' % path)
    return string
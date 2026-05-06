def _get_data(filenames):
    """Read data from file(s) or STDIN.

    Args:
        filenames (list): List of files to read to get data. If empty or
            None, read from STDIN.
    """
    if filenames:
        data = ""
        for filename in filenames:
            with open(filename, "rb") as f:
                data += f.read()
    else:
        data = sys.stdin.read()

    return data
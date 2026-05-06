def _get_python_version_string():
    """
    Returns a string representation of the Python version.

    :return: "2.7.8" if python version is 2.7.8.
    :rtype string
    """
    version_info = sys.version_info
    return '.'.join(map(str, [version_info[0], version_info[1], version_info[2]]))
def get_full_python_version():
    """
    Get full Python version.

    E.g.
        - `2.7.11.final.0.32bit`
        - `3.5.1.final.0.64bit`

    :return: Full Python version.
    """
    # Get version part, e.g. `3.5.1.final.0`
    version_part = '.'.join(str(x) for x in sys.version_info)

    # Get integer width, e.g. 32 or 64
    int_width = struct.calcsize('P') * 8

    # Get integer width part, e.g. `64bit` or `32bit`
    int_width_part = str(int_width) + 'bit'

    # Return full Python version
    return version_part + '.' + int_width_part
def export_file(file_path):
    """Prepend the given parameter with ``export``"""

    if not os.path.isfile(file_path):
        return error("Referenced file does not exist: '{}'.".format(file_path))

    return "export {}".format(file_path)
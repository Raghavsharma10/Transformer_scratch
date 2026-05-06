def read_data(file_path):
    """
    Reads a file and returns a json encoded representation of the file.
    """

    if not is_valid(file_path):
        write_data(file_path, {})

    db = open_file_for_reading(file_path)
    content = db.read()

    obj = decode(content)

    db.close()

    return obj
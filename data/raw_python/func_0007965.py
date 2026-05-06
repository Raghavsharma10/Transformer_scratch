def write_data(path, obj):
    """
    Writes to a file and returns the updated file content.
    """
    with open_file_for_writing(path) as db:
        db.write(encode(obj))

    return obj
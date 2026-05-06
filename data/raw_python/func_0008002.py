def information(filename):
    """Returns the file exif"""
    check_if_this_file_exist(filename)
    filename = os.path.abspath(filename)
    result = get_json(filename)
    result = result[0]
    return result
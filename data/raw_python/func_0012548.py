def safe_filename(filename, replacement="_"):
    """
    Replace unsafe filename characters with underscores. Note that this does not
    test for "legal" names accepted, but a more restricted set of:
    Letters, numbers, spaces, hyphens, underscores and periods.

    :param filename: name of a file as a string
    :param replacement: character to use as a replacement of bad characters
    :return: safe filename string
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    if regex.path.linux.filename.search(filename):
        return filename
    safe_name = ""
    for char in filename:
        safe_name += char if regex.path.linux.filename.search(char) \
            else replacement
    return safe_name
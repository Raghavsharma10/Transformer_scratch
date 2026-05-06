def _get_file_name(file_path):
    """Returns the name of the file at the specified file path, formatted as a UTF-8 bytearray
       terminated with a null character.
    """

    file_name = basename(file_path)

    utf8_file_name = bytearray(file_name, "utf8")
    utf8_file_name.append(0)

    return utf8_file_name
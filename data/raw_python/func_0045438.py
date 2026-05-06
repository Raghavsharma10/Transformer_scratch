def _get_first_64k_content(file_path):
    """Returns the first 65536 (or the file size, whichever is smaller) bytes of the file at the
       specified file path, as a bytearray.
    """

    if not isfile(file_path):
        raise PathDoesNotExistException(file_path)

    file_size = getsize(file_path)

    content_size = min(file_size, 0x10000)

    content = bytearray(content_size)
    with open(file_path, "rb") as file_object:
        content_read = file_object.readinto(content)

        if content_read is None or content_read < content_size:
            raise FileContentReadException(content_size, content_read)

    return content
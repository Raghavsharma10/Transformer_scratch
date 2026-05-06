def _get_file_size(file_path):
    """Returns the size of the file at the specified file path, formatted as a 4-byte unsigned
       integer bytearray.
    """

    size = getsize(file_path)

    file_size = bytearray(4)
    pack_into(b"I", file_size, 0, size)

    return file_size
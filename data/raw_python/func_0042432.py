def get_md5_hash(file_path):
    """
    Calculate the MD5 checksum for a file.

    :param string file_path:
        Path to the file
    :return:
        MD5 checksum
    """
    checksum = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * checksum.block_size), b''):
            checksum.update(chunk)
    return checksum.hexdigest()
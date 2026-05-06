def hash_file(file_path, block_size = 65536):
    """ Hashes a file with sha256 """
    sha = hashlib.sha256()
    with open(file_path, 'rb') as h_file:
        file_buffer = h_file.read(block_size)
        while len(file_buffer) > 0:
            sha.update(file_buffer)
            file_buffer = h_file.read(block_size)
    return sha.hexdigest()
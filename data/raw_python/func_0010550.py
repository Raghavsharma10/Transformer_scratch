def write_stream(src_file, destination_path):
    """
    Write the file-like src_file object to the string dest_path
    :param src_file: file-like data to be written
    :param destination_path: string of the destionation file
    """
    with open(destination_path, 'wb') as destination_file:
        shutil.copyfileobj(fsrc=src_file, fdst=destination_file)
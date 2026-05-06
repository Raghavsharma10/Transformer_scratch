def copy_remote_file(web_file, destination):
    """
    Check if exist the destination path, and copy the online resource
    file to local.

    Args:
        :web_file: reference to online file resource to take.
        :destination: path to store the file.
    """
    size = 0
    dir_name = os.path.dirname(destination)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(destination, 'wb') as file_:
        chunk_size = 8 * 1024
        for chunk in web_file.iter_content(chunk_size=chunk_size):
            if chunk:
                file_.write(chunk)
                size += len(chunk)
    return size
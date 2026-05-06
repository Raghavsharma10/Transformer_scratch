def download_extract_tar(tar_url, folder, tar_filename=''):
    """
    Download and extract the tar at the url to the given folder

    Args:
        tar_url (str): URL of tar file to download
        folder (str): Location of parent directory to extract to. Doesn't have to exist
        tar_filename (str): Location to download tar. Default is to a temp file
    """
    try:
        makedirs(folder)
    except OSError:
        if not isdir(folder):
            raise
    data_file = tar_filename
    if not data_file:
        fd, data_file = mkstemp('.tar.gz')
        download(tar_url, os.fdopen(fd, 'wb'))
    else:
        download(tar_url, data_file)

    with tarfile.open(data_file) as tar:
        tar.extractall(path=folder)
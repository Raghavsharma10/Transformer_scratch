def download(path, source_url):
    """
    Download a file to a given path from a given URL, if it does not exist.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_url: str
        The URL from which to download the file

    Returns
    -------
    str
        The path of the file
    """
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(path):
        print('Downloading {} to {}'.format(source_url, path))
        filename = source_url.split('/')[-1]

        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading {} {:.2%}'.format(
                filename, float(count * block_size) / float(total_size)))
            sys.stdout.flush()
        try:
            urlretrieve(source_url, path, reporthook=_progress)
        except:
            sys.stdout.write('\r')
            # Exception; remove any partially downloaded file and re-raise
            if os.path.exists(path):
                os.remove(path)
            raise
        sys.stdout.write('\r')
    return path
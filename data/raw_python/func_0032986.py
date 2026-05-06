def file_exists(
        download_url,
        filename=None,
        decompress=False,
        subdir=None):
    """
    Return True if a local file corresponding to these arguments
    exists.
    """
    filename = build_local_filename(download_url, filename, decompress)
    full_path = build_path(filename, subdir)
    return os.path.exists(full_path)
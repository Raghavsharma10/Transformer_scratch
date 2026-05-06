def install_package(tar_url, folder, md5_url='{tar_url}.md5',
                    on_download=lambda: None, on_complete=lambda: None):
    """
    Install or update a tar package that has an md5

    Args:
        tar_url (str): URL of package to download
        folder (str): Location to extract tar. Will be created if doesn't exist
        md5_url (str): URL of md5 to use to check for updates
        on_download (Callable): Function that gets called when downloading a new update
        on_complete (Callable): Function that gets called when a new download is complete

    Returns:
        bool: Whether the package was updated
    """
    data_file = join(folder, basename(tar_url))

    md5_url = md5_url.format(tar_url=tar_url)
    try:
        remote_md5 = download(md5_url).decode('utf-8').split(' ')[0]
    except (UnicodeDecodeError, URLError):
        raise ValueError('Invalid MD5 url: ' + md5_url)
    if remote_md5 != calc_md5(data_file):
        on_download()
        if isfile(data_file):
            try:
                with tarfile.open(data_file) as tar:
                    for i in reversed(list(tar)):
                        try:
                            os.remove(join(folder, i.path))
                        except OSError:
                            pass
            except (OSError, EOFError):
                pass

        download_extract_tar(tar_url, folder, data_file)
        on_complete()
        if remote_md5 != calc_md5(data_file):
            raise ValueError('MD5 url does not match tar: ' + md5_url)
        return True
    return False
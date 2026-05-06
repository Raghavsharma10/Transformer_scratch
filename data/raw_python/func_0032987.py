def fetch_file(
        download_url,
        filename=None,
        decompress=False,
        subdir=None,
        force=False,
        timeout=None,
        use_wget_if_available=False):
    """
    Download a remote file and store it locally in a cache directory. Don't
    download it again if it's already present (unless `force` is True.)

    Parameters
    ----------
    download_url : str
        Remote URL of file to download.

    filename : str, optional
        Local filename, used as cache key. If omitted, then determine the local
        filename from the URL.

    decompress : bool, optional
        By default any file whose remote extension is one of (".zip", ".gzip")
        and whose local filename lacks this suffix is decompressed. If a local
        filename wasn't provided but you still want to decompress the stored
        data then set this option to True.

    subdir : str, optional
        Group downloads in a single subdirectory.

    force : bool, optional
        By default, a remote file is not downloaded if it's already present.
        However, with this argument set to True, it will be overwritten.

    timeout : float, optional
        Timeout for download in seconds, default is None which uses
        global timeout.

    use_wget_if_available: bool, optional
        If the `wget` command is available, use that for download instead
        of Python libraries (default True)

    Returns the full path of the local file.
    """
    filename = build_local_filename(download_url, filename, decompress)
    full_path = build_path(filename, subdir)
    if not os.path.exists(full_path) or force:
        logger.info("Fetching %s from URL %s", filename, download_url)
        _download_and_decompress_if_necessary(
            full_path=full_path,
            download_url=download_url,
            timeout=timeout,
            use_wget_if_available=use_wget_if_available)
    else:
        logger.info("Cached file %s from URL %s", filename, download_url)
    return full_path
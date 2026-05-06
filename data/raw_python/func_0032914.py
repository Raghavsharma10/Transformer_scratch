def build_local_filename(download_url=None, filename=None, decompress=False):
    """
    Determine which local filename to use based on the file's source URL,
    an optional desired filename, and whether a compression suffix needs
    to be removed
    """
    assert download_url or filename, "Either filename or URL must be specified"

    # if no filename provided, use the original filename on the server
    if not filename:
        digest = hashlib.md5(download_url.encode('utf-8')).hexdigest()
        parts = split(download_url)
        filename = digest + "." + "_".join(parts)

    filename = normalize_filename(filename)

    if decompress:
        (base, ext) = splitext(filename)
        if ext in (".gz", ".zip"):
            filename = base

    return filename
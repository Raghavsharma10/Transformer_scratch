def download(url, filename=None, print_progress=0, delete_fail=True,
             **kwargs):
    """
    Download a file, optionally printing a simple progress bar
    url: The URL to download
    filename: The filename to save to, default is to use the URL basename
    print_progress: The length of the progress bar, use 0 to disable
    delete_fail: If True delete the file if the download was not successful,
      default is to keep the temporary file
    return: The downloaded filename
    """
    blocksize = 1024 * 1024
    downloaded = 0
    progress = None

    log.info('Downloading %s', url)
    response = open_url(url, **kwargs)

    if not filename:
        filename = os.path.basename(url)

    output = None
    try:
        total = int(response.headers['Content-Length'])

        if print_progress:
            progress = ProgressBar(print_progress, total)

        with tempfile.NamedTemporaryFile(
                prefix=filename + '.', dir='.', delete=False) as output:
            while downloaded < total:
                block = response.read(blocksize)
                output.write(block)
                downloaded += len(block)
                if progress:
                    progress.update(downloaded)
        os.rename(output.name, filename)
        output = None
        return filename
    finally:
        response.close()
        if delete_fail and output:
            os.unlink(output.name)
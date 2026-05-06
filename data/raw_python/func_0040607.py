def download_file(url, destination, **kwargs):
    """
    Download file  process:
        - Open the url
        - Check if it has been downloaded and it hanged.
        - Download it to  the destination folder.

    Args:
        :urls: url to take the file.
        :destionation: place to store the downloaded file.
    """
    web_file = open_remote_url(url, **kwargs)
    file_size = 0

    if not web_file:
        logger.error(
            "Remote file not found. Attempted URLs: {}".format(url))
        return

    modified = is_remote_file_modified(web_file, destination)
    if modified:
        logger.info("Downloading: " + web_file.url)
        file_size = copy_remote_file(web_file, destination)
    else:
        logger.info("File up-to-date: " + destination)

    web_file.close()
    return file_size
def download(url, file=None):
    """
    Pass file as a filename, open file object, or None to return the request bytes

    Args:
        url (str): URL of file to download
        file (Union[str, io, None]): One of the following:
             - Filename of output file
             - File opened in binary write mode
             - None: Return raw bytes instead

    Returns:
        Union[bytes, None]: Bytes of file if file is None
    """
    import urllib.request
    import shutil
    if isinstance(file, str):
        file = open(file, 'wb')
    try:
        with urllib.request.urlopen(url) as response:
            if file:
                shutil.copyfileobj(response, file)
            else:
                return response.read()
    finally:
        if file:
            file.close()
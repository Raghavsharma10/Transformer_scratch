def _chunk_read(response, local_file, chunk_size=65536, initial_size=0):
    """Download a file chunk by chunk and show advancement

    Can also be used when resuming downloads over http.

    Parameters
    ----------
    response: urllib.response.addinfourl
        Response to the download request in order to get file size.
    local_file: file
        Hard disk file where data should be written.
    chunk_size: integer, optional
        Size of downloaded chunks. Default: 8192
    initial_size: int, optional
        If resuming, indicate the initial size of the file.
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py

    bytes_so_far = initial_size
    # Returns only amount left to download when resuming, not the size of the
    # entire file
    total_size = int(response.headers['Content-Length'].strip())
    total_size += initial_size

    progress = ProgressBar(total_size, initial_value=bytes_so_far,
                           max_chars=40, spinner=True, mesg='downloading')
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)
        if not chunk:
            sys.stderr.write('\n')
            break
        _chunk_write(chunk, local_file, progress)
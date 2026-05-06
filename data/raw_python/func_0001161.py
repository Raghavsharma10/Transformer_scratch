def download_file(config, local_path, url, expected_size, chunk_size, log):
    """Download a file.

    :param dict config: Dictionary from get_arguments().
    :param str local_path: Destination path to save file to.
    :param str url: URL of the file to download.
    :param int expected_size: Expected file size in bytes.
    :param int chunk_size: Number of bytes to read in memory before writing to disk and printing a dot.
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.
    """
    if not os.path.exists(os.path.dirname(local_path)):
        log.debug('Creating directory: %s', os.path.dirname(local_path))
        os.makedirs(os.path.dirname(local_path))
    if os.path.exists(local_path):
        log.error('File already exists: %s', local_path)
        raise HandledError
    relative_path = os.path.relpath(local_path, config['dir'] or os.getcwd())
    print(' => {0}'.format(relative_path), end=' ', file=sys.stderr)

    # Download file.
    log.debug('Writing to: %s', local_path)
    with open(local_path, 'wb') as handle:
        response = requests.get(url, stream=True)
        for chunk in response.iter_content(chunk_size):
            handle.write(chunk)
            print('.', end='', file=sys.stderr)

    file_size = os.path.getsize(local_path)
    print(' {0} bytes'.format(file_size), file=sys.stderr)
    if file_size != expected_size:
        log.error('Expected %d bytes but got %d bytes instead.', expected_size, file_size)
        raise HandledError
def download_file(download_url, target_filepath, max_bytes=MAX_FILE_DEFAULT):
    """
    Download a file.

    :param download_url: This field is the url from which data will be
        downloaded.
    :param target_filepath: This field is the path of the file where
        data will be downloaded.
    :param max_bytes: This field is the maximum file size to download. Its
        default value is 128m.
    """
    response = requests.get(download_url, stream=True)
    size = int(response.headers['Content-Length'])

    if _exceeds_size(size, max_bytes, target_filepath) is True:
        return response

    logging.info('Downloading {} ({})'.format(
        target_filepath, format_size(size)))

    if os.path.exists(target_filepath):
        stat = os.stat(target_filepath)
        if stat.st_size == size:
            logging.info('Skipping, file exists and is the right '
                         'size: {}'.format(target_filepath))
            return response
        else:
            logging.info('Replacing, file exists and is the wrong '
                         'size: {}'.format(target_filepath))
            os.remove(target_filepath)

    with open(target_filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logging.info('Download complete: {}'.format(target_filepath))
    return response
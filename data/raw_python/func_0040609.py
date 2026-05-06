def is_remote_file_modified(web_file, destination):
    """
    Check if online file has been modified.
    Args:
        :web_file: online file to check.
        :destination: path of the offline file to compare.
    """
    try:
        # check datetime of last modified in file.
        last_mod = web_file.headers.get('last-modified')
        if last_mod:
            web_file_time = time.strptime(
                web_file.headers.get(
                    'last-modified'), '%a, %d %b %Y %H:%M:%S %Z')
        else:
            web_file_time = time.gmtime()

        web_file_size = int(web_file.headers.get('content-length', -1))
        if os.path.exists(destination):
            file_time = time.gmtime(os.path.getmtime(destination))
            file_size = os.path.getsize(destination)
            if file_time >= web_file_time and file_size == web_file_size:
                return False

    except Exception as ex:
        msg = ('Fail checking if remote file is modified default returns TRUE'
               ' - {}'.format(ex))
        logger.debug(msg)

    return True
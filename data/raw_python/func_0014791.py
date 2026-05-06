def File(url, settings, retry=False):
    """Factory method
    """
    parsed_url = _urlparse(url)

    if parsed_url.scheme == 'gs':
        return GoogleStorageFile(url, settings, retry=retry)
    elif parsed_url.scheme == 'file':
        if parsed_url.hostname == 'localhost' or parsed_url.hostname is None:
            return LocalFile(url, settings, retry=retry)
        else:
            raise FileUtilsError(
                "Cannot process file url %s. Remote file hosts not supported."
                % url)
    else:
        raise FileUtilsError('Unsupported scheme "%s" in file "%s"'
                        % (parsed_url.scheme, url))
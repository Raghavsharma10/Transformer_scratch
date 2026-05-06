def open_remote_url(urls, **kwargs):
    """Open the url and check that it stores a file.
    Args:
        :urls: Endpoint to take the file
    """
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        try:
            web_file = requests.get(url, stream=True, **kwargs)
            if 'html' in web_file.headers['content-type']:
                raise ValueError("HTML source file retrieved.")
            return web_file
        except Exception as ex:
            logger.error('Fail to open remote url - {}'.format(ex))
            continue
def download_raw(url, local_path, callback):
    """
    Download an url to a local file.
    :param url: url of the file to download
    :param local_path: path where the downloaded file should be saved
    :param callback: instance of ProgressCallback
    :return: True is succeeded
    """
    log.debug('download_raw(url={url}, local_path={local_path})'.format(url=url, local_path=local_path))
    raw_progress = RawDownloadProgress(callback)
    reporthook = raw_progress.get_report_hook()
    try:
        log.debug('urlretrieve(url={url}, local_path={local_path}) ...'.format(url=url, local_path=local_path))
        urlretrieve(url=url, filename=local_path, reporthook=reporthook)
        log.debug('... SUCCEEDED')
        callback.finish(True)
        return True
    except URLError:
        log.exception('... FAILED')
        callback.finish(False)
        return False
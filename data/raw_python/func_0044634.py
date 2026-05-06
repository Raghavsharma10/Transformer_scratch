def _download_vswhere():
    """
    Download vswhere to DOWNLOAD_PATH.
    """
    print('downloading from', _get_latest_release_url())
    try:
        from urllib.request import urlopen
        with urlopen(_get_latest_release_url()) as response, open(DOWNLOAD_PATH, 'wb') as outfile:
            shutil.copyfileobj(response, outfile)
    except ImportError:
        # Python 2
        import urllib
        urllib.urlretrieve(_get_latest_release_url(), DOWNLOAD_PATH)
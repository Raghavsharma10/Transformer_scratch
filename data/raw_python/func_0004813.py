def download_get(key, path=".", **kwargs):
    """
    Get a download from GBIF.

    :param key: [str] A key generated from a request, like that from ``download``
    :param path: [str] Path to write zip file to. Default: ``"."``, with a ``.zip`` appended to the end.
    :param **kwargs**: Further named arguments passed on to ``requests.get``

    Downloads the zip file to a directory you specify on your machine.
    The speed of this function is of course proportional to the size of the
    file to download, and affected by your internet connection speed.

    This function only downloads the file. To open and read it, see
    https://github.com/BelgianBiodiversityPlatform/python-dwca-reader

    Usage::

      from pygbif import occurrences as occ
      occ.download_get("0000066-140928181241064")
      occ.download_get("0003983-140910143529206")
    """
    meta = pygbif.occurrences.download_meta(key)
    if meta['status'] != 'SUCCEEDED':
        raise Exception('download "%s" not of status SUCCEEDED' % key)
    else:
        print('Download file size: %s bytes' % meta['size'])
        url = 'http://api.gbif.org/v1/occurrence/download/request/' + key
        path = "%s/%s.zip" % (path, key)
        gbif_GET_write(url, path, **kwargs)
        print("On disk at " + path)
        return {'path': path, 'size': meta['size'], 'key': key}
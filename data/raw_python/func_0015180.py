def download_dap(name, version='', d='', directory=''):
    '''Download a dap to a given or temporary directory
    Return a path to that file together with information if the directory should be later deleted
    '''
    if not d:
        m, d = _get_metadap_dap(name, version)
    if directory:
        _dir = directory
    else:
        _dir = tempfile.mkdtemp()
    try:
        url = d['download']
    except TypeError:
        raise DapiCommError('DAP {dap} has no version to download.'.format(dap=name))
    filename = url.split('/')[-1]
    path = os.path.join(_dir, filename)
    urllib.request.urlretrieve(url, path)
    dapisum = d['sha256sum']
    downloadedsum = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if dapisum != downloadedsum:
        os.remove(path)
        raise DapiLocalError(
            'DAP {dap} has incorrect sha256sum (DAPI: {dapi}, downloaded: {downloaded})'.
            format(dap=name, dapi=dapisum, downloaded=downloadedsum))
    return path, not bool(directory)
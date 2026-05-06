def get_raw_data_from_url(pdb_id, reduced=False):
    """" Get the msgpack unpacked data given a PDB id.

    :param pdb_id: the input PDB id
    :return the unpacked data (a dict) """
    url = get_url(pdb_id,reduced)
    request = urllib2.Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urllib2.urlopen(request)
    if response.info().get('Content-Encoding') == 'gzip':
        data = ungzip_data(response.read())
    else:
        data = response.read()
    return _unpack(data)
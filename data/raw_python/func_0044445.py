def read_json_file(fpath):
    """
    Read a JSON file from ``fpath``; raise an exception if it doesn't exist.

    :param fpath: path to file to read
    :type fpath: str
    :return: deserialized JSON
    :rtype: dict
    """
    if not os.path.exists(fpath):
        raise Exception('ERROR: file %s does not exist.' % fpath)
    with open(fpath, 'r') as fh:
        raw = fh.read()
    res = json.loads(raw)
    return res
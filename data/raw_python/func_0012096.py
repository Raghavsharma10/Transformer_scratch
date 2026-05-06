def write_json(data, filename, gzip_mode=False):
    '''Write the python data structure as a json-Object to filename.'''

    open_file = open
    if gzip_mode:
        open_file = gzip.open

    try:
        with open_file(filename, 'wt') as fh:
            json.dump(obj=data, fp=fh, sort_keys=True)
    except AttributeError:
        # Python-2.6
        fh = open_file(filename, 'wt')
        json.dump(obj=data, fp=fh, sort_keys=True)
        fh.close()
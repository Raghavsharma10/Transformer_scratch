def load_json(filename, gzip_mode=False):
    '''Return the json-file data, with all strings utf-8 encoded.'''

    open_file = open
    if gzip_mode:
        open_file = gzip.open

    try:
        with open_file(filename, 'rt') as fh:
            data = json.load(fh)
            data = convert_unicode_2_utf8(data)
            return data
    except AttributeError:
        # Python-2.6
        fh = open_file(filename, 'rt')
        data = json.load(fh)
        fh.close()
        data = convert_unicode_2_utf8(data)
        return data
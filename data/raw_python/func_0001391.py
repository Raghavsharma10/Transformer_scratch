def read(path, mode='tsv'):
    ''' Helper function to read Document in TTL-TXT format (i.e. ${docname}_*.txt)
    E.g. read('~/data/myfile') is the same as Document('myfile', '~/data/').read()
    '''
    if mode == 'tsv':
        return TxtReader.from_path(path).read()
    elif mode == 'json':
        return read_json(path)
    else:
        raise Exception("Invalid mode - [{}] was provided".format(mode))
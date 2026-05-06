def dumps(columns):
    '''
    Serialize ``columns`` to a JSON formatted ``bytes`` object.
    '''
    fp = BytesIO()
    dump(columns, fp)
    fp.seek(0)
    return fp.read()
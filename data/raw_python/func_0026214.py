def load(fp):
    '''
    Deserialize ``fp`` (a ``.read()``-supporting file-like object
    containing a XPORT document) to a Python object.
    '''
    reader = reading.Reader(fp)
    keys = reader.fields
    columns = {k: [] for k in keys}
    for row in reader:
        for key, value in zip(keys, row):
            columns[key].append(value)
    return columns
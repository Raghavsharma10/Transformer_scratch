def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.
    And decompresses the data with blosc

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        array = dct['__ndarray__']
        # http://stackoverflow.com/questions/24369666/typeerror-b1-is-not-json-serializable
        if sys.version_info >= (3, 0):
            array = array.encode('utf-8')
        data = base64.b64decode(array)
        if has_blosc:
            data = blosc.decompress(data)

        try:
            dtype = np.dtype(ast.literal_eval(dct['dtype']))
        except ValueError:  # If the array is not a recarray
            dtype = dct['dtype']

        return np.frombuffer(data, dtype).reshape(dct['shape'])

    return dct
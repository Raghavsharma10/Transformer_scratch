def dump(pif, fp, **kwargs):
    """
    Convert a single Physical Information Object, or a list of such objects, into a JSON-encoded text file.

    :param pif: Object or list of objects to serialize.
    :param fp: File-like object supporting .write() method to write the serialized object(s) to.
    :param kwargs: Any options available to json.dump().
    """
    return json.dump(pif, fp, cls=PifEncoder, **kwargs)
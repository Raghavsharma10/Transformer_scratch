def data_file(file_fmt, info=None, **kwargs):
    """
    Data file name for given infomation

    Args:
        file_fmt: file format in terms of f-strings
        info: dict, to be hashed and then pass to f-string using 'hash_key'
              these info will also be passed to f-strings
        **kwargs: arguments for f-strings

    Returns:
        str: data file name
    """
    if isinstance(info, dict):
        kwargs['hash_key'] = hashlib.sha256(json.dumps(info).encode('utf-8')).hexdigest()
        kwargs.update(info)

    return utils.fstr(fmt=file_fmt, **kwargs)
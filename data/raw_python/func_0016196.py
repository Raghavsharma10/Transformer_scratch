def json_dumps(obj,       # type: Any
               **kwargs   # type: Any
              ):  # type: (...) -> str
    """ Force use of unicode. """
    if six.PY2:
        kwargs['encoding'] = 'utf-8'
    return json.dumps(convert_to_dict(obj), **kwargs)
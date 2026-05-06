def json_dump(obj,       # type: Any
              fp,        # type: IO[str]
              **kwargs   # type: Any
             ):  # type: (...) -> None
    """ Force use of unicode. """
    if six.PY2:
        kwargs['encoding'] = 'utf-8'
    json.dump(convert_to_dict(obj), fp, **kwargs)
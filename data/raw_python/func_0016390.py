def pack(field: str, kwargs: Dict[str, Any],
         default: Optional[Any] = None, sep: str=',') -> str:
    """ Util for joining multiple fields with commas """
    if default is not None:
        value = kwargs.get(field, default)
    else:
        value = kwargs[field]
    if isinstance(value, str):
        return value
    elif isinstance(value, collections.abc.Iterable):
        return sep.join(str(f) for f in value)
    else:
        return str(value)
def fmtstr(string, *args, **kwargs):
    # type: (Union[Text, bytes, FmtStr], *Any, **Any) -> FmtStr
    """
    Convenience function for creating a FmtStr

    >>> fmtstr('asdf', 'blue', 'on_red', 'bold')
    on_red(bold(blue('asdf')))
    >>> fmtstr('blarg', fg='blue', bg='red', bold=True)
    on_red(bold(blue('blarg')))
    """
    atts = parse_args(args, kwargs)
    if isinstance(string, FmtStr):
        pass
    elif isinstance(string, (bytes, unicode)):
        string = FmtStr.from_str(string)
    else:
        raise ValueError("Bad Args: %r (of type %s), %r, %r" % (string, type(string), args, kwargs))
    return string.copy_with_new_atts(**atts)
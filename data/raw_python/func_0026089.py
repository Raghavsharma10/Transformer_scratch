def format_stack(skip=0, length=6, _sep=os.path.sep):
    """
    Returns a one-line string with the current callstack.
    """
    return ' < '.join("%s:%s:%s" % (
        '/'.join(f.f_code.co_filename.split(_sep)[-2:]),
        f.f_lineno,
        f.f_code.co_name
    ) for f in islice(frame_iterator(sys._getframe(1 + skip)), length))
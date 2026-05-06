def messages(fp, key='@message'):
    """
    Read lines of UTF-8 from the file-like object given in ``fp``, with the
    same fault-tolerance as :function:`tagalog.io.lines`, but instead yield
    dicts with the line data stored in the key given by ``key`` (default:
    "@message").
    """
    for line in lines(fp):
        txt = line.rstrip('\n')
        yield {key: txt}
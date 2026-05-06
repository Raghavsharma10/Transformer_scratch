def split_filename(name):
    """
    Splits the filename into three parts: the name part, the hash part, and the
    extension. Like with the extension, the hash part starts with a dot.

    """
    parts = hashed_filename_re.match(name).groupdict()
    return (parts['name'] or '', parts['hash'] or '', parts['ext'] or '')
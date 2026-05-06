def normalize_filename(filename):
    """
    Remove special characters and shorten if name is too long
    """
    # if the url pointed to a directory then just replace all the special chars
    filename = re.sub("/|\\|;|:|\?|=", "_", filename)

    if len(filename) > 150:
        prefix = hashlib.md5(filename).hexdigest()
        filename = prefix + filename[-140:]

    return filename
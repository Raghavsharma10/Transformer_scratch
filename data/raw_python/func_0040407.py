def text_opener(path, pattern='', verbose=False):
    """Opener that opens single text file.

    :param str path: Path.
    :param str pattern: Regular expression pattern.
    :return: Filehandle(s).
    """
    source = path if is_url(path) else os.path.abspath(path)
    filename = os.path.basename(path)

    if pattern and not re.match(pattern, filename):
        logger.verbose('Skipping file: {}, did not match regex pattern "{}"'.format(os.path.abspath(path), pattern))
        return

    filehandle = urlopen(path) if is_url(path) else open(path)
    logger.verbose('Processing file: {}'.format(source))
    yield filehandle
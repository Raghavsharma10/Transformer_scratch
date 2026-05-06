def filehandles(path, openers_list=openers, pattern='', verbose=False):
    """Main function that iterates over list of openers and decides which opener to use.

    :param str path: Path.
    :param list openers_list: List of openers.
    :param str pattern: Regular expression pattern.
    :param verbose: Print additional information.
    :type verbose: :py:obj:`True` or :py:obj:`False`
    :return: Filehandle(s).
    """
    if not verbose:
        logging.disable(logging.VERBOSE)

    for opener in openers_list:
        try:
            for filehandle in opener(path=path, pattern=pattern, verbose=verbose):
                with closing(filehandle):
                    yield filehandle
            break  # use the first successful opener function

        except (zipfile.BadZipfile, tarfile.ReadError, GZValidationError,
                BZ2ValidationError, IOError, NotADirectoryError):
             continue

        else:
            logger.verbose('No opener found for path: "{}"'.format(path))
            yield None
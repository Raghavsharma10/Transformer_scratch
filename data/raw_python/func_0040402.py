def directory_opener(path, pattern='', verbose=False):
    """Directory opener.

    :param str path: Path.
    :param str pattern: Regular expression pattern.
    :return: Filehandle(s).
    """
    if not os.path.isdir(path):
        raise NotADirectoryError
    else:
        openers_list = [opener for opener in openers if not opener.__name__.startswith('directory')]  # remove directory

        for root, dirlist, filelist in os.walk(path):
            for filename in filelist:

                if pattern and not re.match(pattern, filename):
                    logger.verbose('Skipping file: {}, did not match regex pattern "{}"'.format(os.path.abspath(filename), pattern))
                    continue

                filename_path = os.path.abspath(os.path.join(root, filename))
                for filehandle in filehandles(filename_path, openers_list=openers_list, pattern=pattern, verbose=verbose):
                    yield filehandle
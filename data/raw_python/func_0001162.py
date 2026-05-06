def mangle_coverage(local_path, log):
    """Edit .coverage file substituting Windows file paths to Linux paths.

    :param str local_path: Destination path to save file to.
    :param logging.Logger log: Logger for this function. Populated by with_log() decorator.
    """
    # Read the file, or return if not a .coverage file.
    with open(local_path, mode='rb') as handle:
        if handle.read(13) != b'!coverage.py:':
            log.debug('File %s not a coverage file.', local_path)
            return
        handle.seek(0)

        # I'm lazy, reading all of this into memory. What could possibly go wrong?
        file_contents = handle.read(52428800).decode('utf-8')  # 50 MiB limit, surely this is enough?

    # Substitute paths.
    for windows_path in set(REGEX_MANGLE.findall(file_contents)):
        unix_relative_path = windows_path.replace(r'\\', '/').split('/', 3)[-1]
        unix_absolute_path = os.path.abspath(unix_relative_path)
        if not os.path.isfile(unix_absolute_path):
            log.debug('Windows path: %s', windows_path)
            log.debug('Unix relative path: %s', unix_relative_path)
            log.error('No such file: %s', unix_absolute_path)
            raise HandledError
        file_contents = file_contents.replace(windows_path, unix_absolute_path)

    # Write.
    with open(local_path, 'w') as handle:
        handle.write(file_contents)
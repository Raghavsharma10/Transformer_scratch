def files_with_ext(extension, directory='.', recursive=False):
    """
    Generator function that will iterate over all files in the specified
    directory and return a path to the files which possess a matching extension.

    You should include the period in your extension, and matching is not case
    sensitive: '.xml' will also match '.XML' and vice versa.

    An empty string passed to extension will match extensionless files.
    """
    if recursive:
        log.info('Recursively searching {0} for files with extension "{1}"'.format(directory, extension))
        for dirname, subdirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirname, filename)
                _root, ext = os.path.splitext(filepath)
                if extension.lower() == ext.lower():
                    yield filepath

    else:
        log.info('Looking in {0} for files with extension:  "{1}"'.format(directory, extension))
        for name in os.listdir(directory):
            filepath = os.path.join(directory, name)
            if not os.path.isfile(filepath):  # Skip non-files
                continue
            _root, ext = os.path.splitext(filepath)
            if extension.lower() == ext.lower():
                yield filepath
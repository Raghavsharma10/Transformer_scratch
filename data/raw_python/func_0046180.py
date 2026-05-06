def list_js_files(dir):
    """
    Generator for all JavaScript files in the directory, recursively

    >>> 'examples/module.js' in list(list_js_files('examples'))
    True

    """
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if is_js_file(filename):
                yield os.path.join(dirpath, filename)
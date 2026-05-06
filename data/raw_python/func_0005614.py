def pick_sdf(filename, directory=None):
    """Returns a full path to the chosen SDF file. The supplied file
    is not expected to contain a recognised SDF extension, this is added
    automatically.
    If a file with the extension `.sdf.gz` or `.sdf` is found the path to it
    (excluding the extension) is returned. If this fails, `None` is returned.

    :param filename: The SDF file basename, whose path is required.
    :type filename: ``str``
    :param directory: An optional directory.
                      If not provided it is calculated automatically.
    :type directory: ``str``
    :return: The full path to the file without extension,
             or None if it does not exist
    :rtype: ``str``
    """
    if directory is None:
        directory = utils.get_undecorated_calling_module()
        # If the 'cwd' is not '/output' (which indicates we're in a Container)
        # then remove the CWD and the anticipated '/'
        # from the front of the module
        if os.getcwd() not in ['/output']:
            directory = directory[len(os.getcwd()) + 1:]

    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path + '.sdf.gz'):
        return file_path + '.sdf.gz'
    elif os.path.isfile(file_path + '.sdf'):
        return file_path + '.sdf'
    # Couldn't find a suitable SDF file
    return None
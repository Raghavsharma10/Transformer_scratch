def _input_directory_description(input_identifier, arg_item, input_dir):
    """
     Produces a directory description. A directory description is a dictionary containing the following information.

     - 'path': An array containing the paths to the specified directories.
     - 'debugInfo': A field to possibly provide debug information.
     - 'found': A boolean that indicates, if the directory exists in the local filesystem.
     - 'listing': A listing that shows which files are in the given directory. This could be None.

    :param input_identifier: The input identifier in the cwl description file
    :param arg_item: The corresponding job information
    :param input_dir: TODO
    :return: A directory description
    :raise DirectoryError: If the given directory does not exist or is not a directory.
    """
    description = {
        'path': None,
        'found': False,
        'debugInfo': None,
        'listing': None,
        'basename': None
    }

    try:
        path = location(input_identifier, arg_item)

        if input_dir and not os.path.isabs(path):
            path = os.path.join(os.path.expanduser(input_dir), path)

        description['path'] = path
        if not os.path.exists(path):
            raise DirectoryError('path does not exist')
        if not os.path.isdir(path):
            raise DirectoryError('path is not a directory')

        description['listing'] = arg_item.get('listing')
        description['basename'] = os.path.basename(path)

        description['found'] = True
    except:
        description['debugInfo'] = exception_format()

    return description
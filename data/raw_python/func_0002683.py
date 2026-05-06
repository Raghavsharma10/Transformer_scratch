def _check_input_directory_listing(base_directory, listing):
    """
    Raises an DirectoryError if files or directories, given in the listing, could not be found in the local filesystem.

    :param base_directory: The path to the directory to check
    :param listing: A listing given as dictionary
    :raise DirectoryError: If the given base directory does not contain all of the subdirectories and subfiles given in
    the listing.
    """

    for sub in listing:
        path = os.path.join(base_directory, sub['basename'])
        if sub['class'] == 'File':
            if not os.path.isfile(path):
                raise DirectoryError('File \'{}\' not found but specified in listing.'.format(path))
        if sub['class'] == 'Directory':
            if not os.path.isdir(path):
                raise DirectoryError('Directory \'{}\' not found but specified in listing'.format(path))
            sub_listing = sub.get('listing')
            if sub_listing:
                _check_input_directory_listing(path, sub_listing)
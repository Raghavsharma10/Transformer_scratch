def pickle_save(data, name, **kwargs):
    """Saves object with pickle.

    Parameters
    ----------
    data: anything picklable
        Object to save.
    name: str
        Path to save to (includes dir, excludes extension).
    extension: str, optional
        File extension.
    overwrite existing: bool, optional
        When the save path already contains file: if True, file will be
        overwritten, if False the data will be saved with the system time
        appended to the file name.
    """
    extension = kwargs.pop('extension', '.pkl')
    overwrite_existing = kwargs.pop('overwrite_existing', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    filename = name + extension
    # Check if the target directory exists and if not make it
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname)
    if os.path.isfile(filename) and not overwrite_existing:
        print(filename + ' already exists! Saving with time appended')
        filename = name + '_' + time.asctime().replace(' ', '_')
        filename += extension
    # check if permission error is defined (was not before python 3.3)
    # and otherwise use IOError
    try:
        PermissionError
    except NameError:
        PermissionError = IOError
    try:
        outfile = open(filename, 'wb')
        pickle.dump(data, outfile)
        outfile.close()
    except (MemoryError, PermissionError) as err:
        warnings.warn((type(err).__name__ + ' in pickle_save: continue without'
                       ' saving.'), UserWarning)
def _exists(filenames):
    """Check if every filename exists. If not, print an error
    message and remove the item from the list.

    Parameters
    ----------
    filenames : list
        List of filenames to check for existence.

    Returns
    -------
    list
        Filtered list of filenames that exists.
    """
    exists = []
    for filename in filenames:
        if os.path.isfile(filename):
            exists.append(filename)
        else:
            print('fijibin ERROR missing output file {}'.format(filename))

    return exists
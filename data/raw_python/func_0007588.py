def validate_subfolders(filedir, metadata):
    """
    Check that all folders in the given directory have a corresponding
    entry in the metadata file, and vice versa.

    :param filedir: This field is the target directory from which to
        match metadata
    :param metadata: This field contains the metadata to be matched.
    """
    if not os.path.isdir(filedir):
        print("Error: " + filedir + " is not a directory")
        return False
    subfolders = os.listdir(filedir)
    for subfolder in subfolders:
        if subfolder not in metadata:
            print("Error: folder " + subfolder +
                  " present on disk but not in metadata")
            return False
    for subfolder in metadata:
        if subfolder not in subfolders:
            print("Error: folder " + subfolder +
                  " present in metadata but not on disk")
            return False
    return True
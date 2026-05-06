def validate_metadata(target_dir, metadata):
    """
    Check that the files listed in metadata exactly match files in target dir.

    :param target_dir: This field is the target directory from which to
        match metadata
    :param metadata: This field contains the metadata to be matched.
    """
    if not os.path.isdir(target_dir):
        print("Error: " + target_dir + " is not a directory")
        return False
    file_list = os.listdir(target_dir)
    for filename in file_list:
        if filename not in metadata:
            print("Error: " + filename + " present at" + target_dir +
                  " not found in metadata file")
            return False
    for filename in metadata:
        if filename not in file_list:
            print("Error: " + filename + " present in metadata file " +
                  " not found on disk at: " + target_dir)
            return False
    return True
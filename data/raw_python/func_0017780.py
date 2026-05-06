def copy_and_verify(path, source_path, sha256):
    """
    Copy a file to a given path from a given path, if it does not exist.
    After copying it, verify it integrity by checking the SHA-256 hash.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_path: str
        The path from which to copy the file
    sha256: str
        The expected SHA-256 hex digest of the file, or `None` to print the
        digest of the file to the console

    Returns
    -------
    str or None
        The path of the file if successfully downloaded otherwise `None`
    """
    if os.path.exists(path):
        # Already exists?
        # Nothing to do, except print the SHA-256 if necessary
        if sha256 is None:
            print('The SHA-256 of {} is "{}"'.format(
                path, compute_sha256(path)))
        return path

    if not os.path.exists(source_path):
        return None

    # Compute the path of the unverified file
    unverified_path = path + '.unverified'
    # Copy it
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    shutil.copy(source_path, unverified_path)

    if os.path.exists(unverified_path):
        # Got something...
        if verify_file(unverified_path, sha256):
            # Success: rename the unverified file to the destination
            # filename
            os.rename(unverified_path, path)
            return path
        else:
            # Report failure
            print('SHA verification of file {} failed'.format(source_path))
            # Delete
            os.remove(unverified_path)
    return None
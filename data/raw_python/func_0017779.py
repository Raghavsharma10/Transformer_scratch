def download_and_verify(path, source_url, sha256):
    """
    Download a file to a given path from a given URL, if it does not exist.
    After downloading it, verify it integrity by checking the SHA-256 hash.

    Parameters
    ----------
    path: str
        The (destination) path of the file on the local filesystem
    source_url: str
        The URL from which to download the file
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

    # Compute the path of the unverified file
    unverified_path = path + '.unverified'
    for i in range(_MAX_DOWNLOAD_TRIES):
        # Download it
        try:
            unverified_path = download(unverified_path, source_url)
        except Exception as e:
            # Report failure
            print(
                'Download of {} unsuccessful; error {}; '
                'deleting and re-trying...'.format(source_url, e))
            # Delete so that we can retry
            if os.path.exists(unverified_path):
                os.remove(unverified_path)
        else:
            if os.path.exists(unverified_path):
                # Got something...
                if verify_file(unverified_path, sha256):
                    # Success: rename the unverified file to the destination
                    # filename
                    os.rename(unverified_path, path)
                    return path
                else:
                    # Report failure
                    print(
                        'Download of {} unsuccessful; verification failed; '
                        'deleting and re-trying...'.format(source_url))
                    # Delete so that we can retry
                    os.remove(unverified_path)

    print('Did not succeed in downloading {} (tried {} times)'.format(
        source_url, _MAX_DOWNLOAD_TRIES
    ))
    return None
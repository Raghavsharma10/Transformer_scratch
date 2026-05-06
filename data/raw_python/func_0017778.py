def verify_file(path, sha256):
    """
    Verify the integrity of a file by checking its SHA-256 hash.
    If no digest is supplied, the digest is printed to the console.

    Closely follows the code in `torchvision.datasets.utils.check_integrity`

    Parameters
    ----------
    path: str
        The path of the file to check
    sha256: str
        The expected SHA-256 hex digest of the file, or `None` to print the
        digest of the file to the console

    Returns
    -------
    bool
        Indicates if the file passes the integrity check or not
    """
    if not os.path.isfile(path):
        return False
    digest = compute_sha256(path)
    if sha256 is None:
        # No digest supplied; report it to the console so a develop can fill
        # it in
        print('SHA-256 of {}:'.format(path))
        print('  "{}"'.format(digest))
    else:
        if digest != sha256:
            return False
    return True
def make_file_read_only(file_path):
    """
    Removes the write permissions for the given file for owner, groups and others.

    :param file_path: The file whose privileges are revoked.
    :raise FileNotFoundError: If the given file does not exist.
    """
    old_permissions = os.stat(file_path).st_mode
    os.chmod(file_path, old_permissions & ~WRITE_PERMISSIONS)
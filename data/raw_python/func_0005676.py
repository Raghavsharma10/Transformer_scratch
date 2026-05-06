def assert_secure_file(file):
    """checks if a file is stored securely"""
    if not is_secure_file(file):
        msg = """
        File {0} can be read by other users.
        This is not secure. Please run 'chmod 600 "{0}"'"""
        raise SecurityError(dedent(msg).replace('\n', ' ').format(file))
    return True
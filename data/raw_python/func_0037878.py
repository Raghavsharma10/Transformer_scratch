def find_file(search_dir, file_pattern):
    """
    Search for a file in a directory, and return the first match.
    If the file is not found return an empty string

    Args:
        search_dir: The root directory to search in
        file_pattern: A unix-style wildcard pattern representing
            the file to find

    Returns:
        The path to the file if it was found, otherwise an empty string
    """
    for root, dirnames, fnames in os.walk(search_dir):
            for fname in fnames:
                if fnmatch.fnmatch(fname, file_pattern):
                    return os.path.join(root, fname)
    return ""
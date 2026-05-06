def find_file_in_load_dirs(relpath):
    """If given relative path exists in one of DevAssistant load paths,
    return its full path.

    Args:
        relpath: a relative path, e.g. "assitants/crt/test.yaml"

    Returns:
        absolute path of the file, e.g. "/home/x/.devassistant/assistanta/crt/test.yaml
        or None if file is not found
    """
    if relpath.startswith(os.path.sep):
        relpath = relpath.lstrip(os.path.sep)

    for ld in settings.DATA_DIRECTORIES:
        possible_path = os.path.join(ld, relpath)
        if os.path.exists(possible_path):
            return possible_path
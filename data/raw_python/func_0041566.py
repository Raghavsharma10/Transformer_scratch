def get_directory_relative_to_git_root(directory: str):
    """
    Gets the path to the given directory relative to the git repository root in which it is a subdirectory.
    :param directory: the directory within a git repository
    :return: the path to the directory relative to the git repository root
    """
    return os.path.relpath(os.path.realpath(directory), get_git_root_directory(directory))
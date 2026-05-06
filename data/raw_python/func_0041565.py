def get_git_root_directory(directory: str):
    """
    Gets the path of the git project root directory from the given directory.
    :param directory: the directory within a git repository
    :return: the root directory of the git repository
    :exception NotAGitRepositoryException: raised if the given directory is not within a git repository
    """
    try:
        return run([GIT_COMMAND, "rev-parse", "--show-toplevel"], directory)
    except RunException as e:
        if " Not a git repository" in e.stderr:
            raise NotAGitRepositoryException(directory) from e
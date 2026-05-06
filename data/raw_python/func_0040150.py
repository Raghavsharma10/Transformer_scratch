def pull(directory: str) -> Commit:
    """
    Pulls the subrepo that has been cloned into the given directory.
    :param directory: the directory containing the subrepo
    :return: the commit the subrepo is on
    """
    if not os.path.exists(directory):
        raise ValueError(f"No subrepo found in \"{directory}\"")
    try:
        result = run([GIT_COMMAND, _GIT_SUBREPO_COMMAND, _GIT_SUBREPO_PULL_COMMAND, _GIT_SUBREPO_VERBOSE_FLAG,
                      get_directory_relative_to_git_root(directory)],
                     execution_directory=get_git_root_directory(directory))
    except RunException as e:
        if "Can't pull subrepo. Working tree has changes" in e.stderr:
            raise UnstagedChangeException() from e
    return status(directory)[2]
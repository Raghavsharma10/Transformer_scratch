def status(directory: str) -> Tuple[RepositoryLocation, Branch, Commit]:
    """
    Gets the status of the subrepo that has been cloned into the given directory.
    :param directory: the directory containing the subrepo
    :return: a tuple consisting of the URL the subrepo is tracking, the branch that has been checked out and the commit
    reference
    """
    if not os.path.exists(directory):
        raise ValueError(f"No subrepo found in \"{directory}\"")

    try:
        result = run([GIT_COMMAND, _GIT_SUBREPO_COMMAND, _GIT_SUBREPO_STATUS_COMMAND, _GIT_SUBREPO_VERBOSE_FLAG,
                      get_directory_relative_to_git_root(directory)],
                     execution_directory=get_git_root_directory(directory))
    except RunException as e:
        if "Command failed: 'git rev-parse --verify HEAD'" in e.stderr:
            raise NotAGitSubrepoException(directory) from e
        raise e

    if re.search("is not a subrepo$", result):
        raise NotAGitSubrepoException(directory)

    url = re.search("Remote URL:\s*(.*)", result).group(1)
    branch = re.search("Tracking Branch:\s*(.*)", result).group(1)
    commit = re.search("Pulled Commit:\s*(.*)", result).group(1)
    return url, branch, commit
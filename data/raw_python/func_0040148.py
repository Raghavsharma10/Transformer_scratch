def clone(location: str, directory: str, *, branch: str=None, tag: str=None, commit: str=None, author_name: str=None,
          author_email: str=None) -> Commit:
    """
    Clones the repository at the given location as a subrepo in the given directory.
    :param location: the location of the repository to clone
    :param directory: the directory that the subrepo will occupy (i.e. not the git repository root)
    :param branch: the specific branch to clone
    :param tag: the specific tag to clone
    :param commit: the specific commit to clone (may also require tag/branch to be specified if not fetched)
    :param author_name: the name of the author to assign to the clone commit (uses system specified if not set)
    :param author_email: the email of the author to assign to the clone commit (uses system specified if not set)
    :return: the commit reference of the checkout
    """
    if os.path.exists(directory):
        raise ValueError(f"The directory \"{directory}\" already exists")
    if not os.path.isabs(directory):
        raise ValueError(f"Directory must be absolute: {directory}")
    if branch and tag:
        raise ValueError(f"Cannot specify both branch \"{branch}\" and tag \"{tag}\"")
    if not branch and not tag and not commit:
        branch = _DEFAULT_BRANCH

    existing_parent_directory = directory
    while not os.path.exists(existing_parent_directory):
        assert existing_parent_directory != ""
        existing_parent_directory = os.path.dirname(existing_parent_directory)

    git_root = get_git_root_directory(existing_parent_directory)
    git_relative_directory = os.path.relpath(os.path.realpath(directory), git_root)

    if (branch or tag) and commit:
        run([GIT_COMMAND, "fetch", location, branch if branch else tag], execution_directory=git_root)
        branch, tag = None, None
    reference = branch if branch else (tag if tag else commit)

    execution_environment = os.environ.copy()
    if author_name is not None:
        execution_environment[_GIT_AUTHOR_NAME_ENVIRONMENT_VARIABLE] = author_name
    if author_email is not None:
        execution_environment[_GIT_AUTHOR_EMAIL_ENVIRONMENT_VARIABLE] = author_email

    try:
        run([GIT_COMMAND, _GIT_SUBREPO_COMMAND, _GIT_SUBREPO_CLONE_COMMAND, _GIT_SUBREPO_VERBOSE_FLAG,
             _GIT_SUBREPO_BRANCH_FLAG, reference, location, git_relative_directory], execution_directory=git_root,
            execution_environment=execution_environment)
    except RunException as e:
        if re.search("Can't clone subrepo. (Unstaged|Index has) changes", e.stderr) is not None:
            raise UnstagedChangeException(git_root) from e
        elif "Command failed:" in e.stderr:
            try:
                repo_info = run([GIT_COMMAND, _GIT_LS_REMOTE_COMMAND, location])
                if not branch and not tag and commit:
                    raise NotAGitReferenceException(
                        f"Commit \"{commit}\" not found (specify branch/tag to fetch that first if required)")
                else:
                    references = re.findall("^.+\srefs\/.+\/(.+)", repo_info, flags=re.MULTILINE)
                    if reference not in references:
                        raise NotAGitReferenceException(f"{reference} not found in {references}") from e

            except RunException as debug_e:
                if re.match("fatal: repository .* not found", debug_e.stderr):
                    raise NotAGitRepositoryException(location) from e
        raise e

    assert os.path.exists(directory)
    return status(directory)[2]
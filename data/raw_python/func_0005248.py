def get_base_branch():
    # type: () -> str
    """ Return the base branch for the current branch.

    This function will first try to guess the base branch and if it can't it
    will let the user choose the branch from the list of all local branches.

    Returns:
        str: The name of the branch the current branch is based on.
    """
    base_branch = git.guess_base_branch()

    if base_branch is None:
        log.info("Can't guess the base branch, you have to pick one yourself:")
        base_branch = choose_branch()

    return base_branch
def git_branch_rename(new_name):
    # type: (str) -> None
    """ Rename the current branch

    Args:
        new_name (str):
            New name for the current branch.
    """
    curr_name = git.current_branch(refresh=True).name

    if curr_name not in git.protected_branches():
        log.info("Renaming branch from <33>{}<32> to <33>{}".format(
            curr_name, new_name
        ))
        shell.run('git branch -m {}'.format(new_name))
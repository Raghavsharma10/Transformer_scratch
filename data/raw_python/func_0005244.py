def git_branch_delete(branch_name):
    # type: (str) -> None
    """ Delete the given branch.

    Args:
        branch_name (str):
            Name of the branch to delete.
    """
    if branch_name not in git.protected_branches():
        log.info("Deleting branch <33>{}", branch_name)
        shell.run('git branch -d {}'.format(branch_name))
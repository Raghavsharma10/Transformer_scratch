def verify_branch(branch_name):
    # type: (str) -> bool
    """ Verify if the given branch exists.

    Args:
        branch_name (str):
            The name of the branch to check.

    Returns:
        bool: **True** if a branch with name *branch_name* exits, **False**
        otherwise.
    """
    try:
        shell.run(
            'git rev-parse --verify {}'.format(branch_name),
            never_pretend=True
        )
        return True
    except IOError:
        return False
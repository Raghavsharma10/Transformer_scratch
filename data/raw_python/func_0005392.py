def update():
    # type: () -> None
    """ Update the feature with updates committed to develop.

    This will merge current develop into the current branch.
    """
    branch = git.current_branch(refresh=True)
    base_branch = common.get_base_branch()

    common.assert_branch_type('task')
    common.git_checkout(base_branch)
    common.git_pull(base_branch)
    common.git_checkout(branch.name)
    common.git_merge(branch.name, base_branch)
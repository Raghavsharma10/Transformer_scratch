def update():
    # type: () -> None
    """ Update the feature with updates committed to develop.

    This will merge current develop into the current branch.
    """
    branch = git.current_branch(refresh=True)
    develop = conf.get('git.devel_branch', 'develop')

    common.assert_branch_type('feature')
    common.git_checkout(develop)
    common.git_pull(develop)
    common.git_checkout(branch.name)
    common.git_merge(branch.name, develop)
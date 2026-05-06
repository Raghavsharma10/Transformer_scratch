def merged():
    # type: () -> None
    """ Cleanup a remotely merged branch. """
    base_branch = common.get_base_branch()
    branch = git.current_branch(refresh=True)

    common.assert_branch_type('task')

    # Pull feature branch with the merged task
    common.git_checkout(base_branch)
    common.git_pull(base_branch)

    # Cleanup
    common.git_branch_delete(branch.name)
    common.git_prune()

    common.git_checkout(base_branch)
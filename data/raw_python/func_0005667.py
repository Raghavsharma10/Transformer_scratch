def merged():
    # type: () -> None
    """ Cleanup a remotely merged branch. """
    develop = conf.get('git.devel_branch', 'develop')
    branch = git.current_branch(refresh=True)

    common.assert_branch_type('feature')

    # Pull develop with the merged feature
    common.git_checkout(develop)
    common.git_pull(develop)

    # Cleanup
    common.git_branch_delete(branch.name)
    common.git_prune()

    common.git_checkout(develop)
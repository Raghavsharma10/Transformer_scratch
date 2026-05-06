def merged():
    # type: () -> None
    """ Cleanup a remotely merged branch. """
    develop = conf.get('git.devel_branch', 'develop')
    master = conf.get('git.master_branch', 'master')
    branch = git.current_branch(refresh=True)

    common.assert_branch_type('hotfix')

    # Pull master with the merged hotfix
    common.git_checkout(master)
    common.git_pull(master)

    # Merge to develop
    common.git_checkout(develop)
    common.git_pull(develop)
    common.git_merge(develop, branch.name)

    # Cleanup
    common.git_branch_delete(branch.name)
    common.git_prune()

    common.git_checkout(master)
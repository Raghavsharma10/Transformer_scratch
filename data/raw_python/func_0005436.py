def finish():
    # type: () -> None
    """ Merge current feature into develop. """
    pretend = context.get('pretend', False)

    if not pretend and (git.staged() or git.unstaged()):
        log.err(
            "You have uncommitted changes in your repo!\n"
            "You need to stash them before you merge the hotfix branch"
        )
        sys.exit(1)

    develop = conf.get('git.devel_branch', 'develop')
    master = conf.get('git.master_branch', 'master')
    branch = git.current_branch(refresh=True)

    common.assert_branch_type('hotfix')

    # Merge hotfix into master
    common.git_checkout(master)
    common.git_pull(master)
    common.git_merge(master, branch.name)

    # Merge hotfix into develop
    common.git_checkout(develop)
    common.git_pull(develop)
    common.git_merge(develop, branch.name)

    # Cleanup
    common.git_branch_delete(branch.name)
    common.git_prune()

    common.git_checkout(master)
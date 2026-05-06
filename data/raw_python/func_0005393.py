def finish():
    # type: () -> None
    """ Merge current feature branch into develop. """
    pretend = context.get('pretend', False)

    if not pretend and (git.staged() or git.unstaged()):
        log.err(
            "You have uncommitted changes in your repo!\n"
            "You need to stash them before you merge the hotfix branch"
        )
        sys.exit(1)

    branch = git.current_branch(refresh=True)
    base = common.get_base_branch()

    prompt = "<32>Merge <33>{}<32> into <33>{}<0>?".format(branch.name, base)
    if not click.confirm(shell.fmt(prompt)):
        log.info("Cancelled")
        return

    common.assert_branch_type('task')

    # Merge task into it's base feature branch
    common.git_checkout(base)
    common.git_pull(base)
    common.git_merge(base, branch.name)

    # Cleanup
    common.git_branch_delete(branch.name)
    common.git_prune()

    common.git_checkout(base)
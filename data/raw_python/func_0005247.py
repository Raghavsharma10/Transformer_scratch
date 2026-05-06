def git_merge(base, head, no_ff=False):
    # type: (str, str, bool) -> None
    """ Merge *head* into *base*.

    Args:
        base (str):
            The base branch. *head* will be merged into this branch.
        head (str):
            The branch that will be merged into *base*.
        no_ff (bool):
            If set to **True** it will force git to create merge commit. If set
            to **False** (default) it will do a fast-forward merge if possible.
    """
    pretend = context.get('pretend', False)
    branch = git.current_branch(refresh=True)

    if branch.name != base and not pretend:
        git_checkout(base)

    args = []

    if no_ff:
        args.append('--no-ff')

    log.info("Merging <33>{}<32> into <33>{}<32>", head, base)
    shell.run('git merge {args} {branch}'.format(
        args=' '.join(args),
        branch=head,
    ))

    if branch.name != base and not pretend:
        git_checkout(branch.name)
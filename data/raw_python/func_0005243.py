def assert_on_branch(branch_name):
    # type: (str) -> None
    """ Print error and exit if *branch_name* is not the current branch.

    Args:
        branch_name (str):
            The supposed name of the current branch.
    """
    branch = git.current_branch(refresh=True)

    if branch.name != branch_name:
        if context.get('pretend', False):
            log.info("Would assert that you're on a <33>{}<32> branch",
                     branch_name)
        else:
            log.err("You're not on a <33>{}<31> branch!", branch_name)
            sys.exit(1)
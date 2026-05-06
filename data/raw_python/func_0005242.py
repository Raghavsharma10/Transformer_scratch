def assert_branch_type(branch_type):
    # type: (str) -> None
    """ Print error and exit if the current branch is not of a given type.

    Args:
        branch_type (str):
            The branch type. This assumes the branch is in the '<type>/<title>`
            format.
    """
    branch = git.current_branch(refresh=True)

    if branch.type != branch_type:
        if context.get('pretend', False):
            log.info("Would assert that you're on a <33>{}/*<32> branch",
                     branch_type)
        else:
            log.err("Not on a <33>{}<31> branch!", branch_type)
            fmt = ("The branch must follow <33>{required_type}/<name><31>"
                   "format and your branch is called <33>{name}<31>.")
            log.err(fmt, required_type=branch_type, name=branch.name)
            sys.exit(1)
def checkout(lancet, force, issue):
    """
    Checkout the branch for the given issue.

    It is an error if the branch does no exist yet.
    """
    issue = get_issue(lancet, issue)

    # Get the working branch
    branch = get_branch(lancet, issue, create=force)

    with taskstatus("Checking out working branch") as ts:
        if not branch:
            ts.abort("Working branch not found")
        lancet.repo.checkout(branch.name)
        ts.ok('Checked out "{}"', branch.name)
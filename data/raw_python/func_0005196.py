def current_branch():
    # type: () -> BranchDetails
    """ Return the BranchDetails for the current branch.

    Return:
        BranchDetails: The details of the current branch.
    """
    cmd = 'git symbolic-ref --short HEAD'
    branch_name = shell.run(
        cmd,
        capture=True,
        never_pretend=True
    ).stdout.strip()

    return BranchDetails.parse(branch_name)
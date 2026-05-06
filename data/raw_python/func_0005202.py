def branches():
    # type: () -> List[str]
    """ Return a list of branches in the current repo.

    Returns:
        list[str]: A list of branches in the current repo.
    """
    out = shell.run(
        'git branch',
        capture=True,
        never_pretend=True
    ).stdout.strip()
    return [x.strip('* \t\n') for x in out.splitlines()]
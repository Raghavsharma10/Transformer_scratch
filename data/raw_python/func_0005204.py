def config():
    # type: () -> dict[str, Any]
    """ Return the current git configuration.

    Returns:
        dict[str, Any]: The current git config taken from ``git config --list``.
    """
    out = shell.run(
        'git config --list',
        capture=True,
        never_pretend=True
    ).stdout.strip()

    result = {}
    for line in out.splitlines():
        name, value = line.split('=', 1)
        result[name.strip()] = value.strip()

    return result
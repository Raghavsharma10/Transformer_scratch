def tags():
    # type: () -> List[str]
    """ Returns all tags in the repo.

    Returns:
        list[str]: List of all tags in the repo, sorted as versions.

    All tags returned by this function will be parsed as if the contained
    versions (using ``v:refname`` sorting).
    """
    return shell.run(
        'git tag --sort=v:refname',
        capture=True,
        never_pretend=True
    ).stdout.strip().splitlines()
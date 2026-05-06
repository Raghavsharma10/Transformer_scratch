def tag(name, message, author=None):
    # type: (str, str, Author, bool) -> None
    """ Tag the current commit.

    Args:
        name (str):
            The tag name.
        message (str):
            The tag message. Same as ``-m`` parameter in ``git tag``.
        author (Author):
            The commit author. Will default to the author of the commit.
        pretend (bool):
            If set to **True** it will print the full ``git tag`` command
            instead of actually executing it.
    """
    cmd = (
        'git -c "user.name={author.name}" -c "user.email={author.email}" '
        'tag -a "{name}" -m "{message}"'
    ).format(
        author=author or latest_commit().author,
        name=name,
        message=message.replace('"', '\\"').replace('`', '\\`'),
    )
    shell.run(cmd)
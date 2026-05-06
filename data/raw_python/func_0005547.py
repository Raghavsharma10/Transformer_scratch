def tag_release(message):
    # type: (str, bool) -> None
    """ Tag the current commit with as the current version release.

    This should be the same commit as the one that's uploaded as the release
    (to pypi for example).

    **Example Config**::

        \b
        version_file: 'src/mypkg/__init__.py'

    Examples::

        $ peltak release tag          # Tag the current commit as release

    """
    from peltak.extra.gitflow import logic
    logic.release.tag(message)
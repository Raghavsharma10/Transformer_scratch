def start(component, exact):
    # type: (str) -> None
    """ Create a new release.

    It will bump the current version number and create a release branch called
    `release/<version>` with one new commit (the version bump).

    **Example Config**::

        \b
        version_file: 'src/mypkg/__init__.py'

    **Examples**::

        \b
        $ peltak release start patch    # Make a new patch release
        $ peltak release start minor    # Make a new minor release
        $ peltak release start major    # Make a new major release
        $ peltak release start          # same as start patch

    """
    from peltak.extra.gitflow import logic
    logic.release.start(component, exact)
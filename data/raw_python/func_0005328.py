def version_cli(ctx, porcelain):
    # type: (click.Context, bool) -> None
    """ Show project version. Has sub commands.

    For this command to work you must specify where the project version is
    stored. You can do that with version_file conf variable. peltak supports
    multiple ways to store the project version. Right now you can store it in a
    python file using built-in __version__ variable. You can use node.js
    package.json and keep the version there or you can just use a plain text
    file that just holds the raw project version. The appropriate storage is
    guessed based on the file type and name.

    Example Configuration::

        version_file: 'src/mypackage/__init__.py'

    Examples:

        \b
        $ peltak version                        # Pretty print current version
        $ peltak version --porcelain            # Print version as raw string
        $ peltak version bump patch             # Bump patch version component
        $ peltak version bump minor             # Bump minor version component
        $ peltak version bump major             # Bump major version component
        $ peltak version bump release           # same as version bump patch
        $ peltak version bump --exact=1.2.1     # Set project version to 1.2.1

    """
    if ctx.invoked_subcommand:
        return

    from peltak.core import log
    from peltak.core import versioning

    current = versioning.current()

    if porcelain:
        print(current)
    else:
        log.info("Version: <35>{}".format(current))
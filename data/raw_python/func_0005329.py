def bump_version(component='patch', exact=None):
    # type: (str, str) -> None
    """ Bump current project version without committing anything.

    No tags are created either.

    Examples:

        \b
        $ peltak version bump patch             # Bump patch version component
        $ peltak version bump minor             # Bump minor version component
        $ peltak version bump major             # Bump major version component
        $ peltak version bump release           # same as version bump patch
        $ peltak version bump --exact=1.2.1     # Set project version to 1.2.1

    """
    from peltak.core import log
    from peltak.core import versioning

    old_ver, new_ver = versioning.bump(component, exact)

    log.info("Project version bumped")
    log.info("  old version: <35>{}".format(old_ver))
    log.info("  new version: <35>{}".format(new_ver))
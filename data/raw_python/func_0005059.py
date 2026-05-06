def tag(message):
    # type: () -> None
    """ Tag the current commit with the current version. """
    release_ver = versioning.current()
    message = message or 'v{} release'.format(release_ver)

    with conf.within_proj_dir():
        log.info("Creating release tag")
        git.tag(
            author=git.latest_commit().author,
            name='v{}'.format(release_ver),
            message=message,
        )
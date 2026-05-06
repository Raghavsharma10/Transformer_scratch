def upload(ctx, yes=False):
    """Upload the package to PyPI."""
    import callee
    version = callee.__version__

    # check the packages version
    # TODO: add a 'release' to automatically bless a version as release one
    if version.endswith('-dev'):
        fatal("Can't upload a development version (%s) to PyPI!", version)

    # run the upload if it has been confirmed by the user
    if not yes:
        answer = input("Do you really want to upload to PyPI [y/N]? ")
        yes = answer.strip().lower() == 'y'
    if not yes:
        logging.warning("Aborted -- not uploading to PyPI.")
        return -2

    logging.debug("Uploading version %s to PyPI...", version)
    setup_py_upload = ctx.run('python setup.py sdist upload')
    if not setup_py_upload.ok:
        fatal("Failed to upload version %s to PyPI!", version,
              cause=setup_py_upload)
    logging.info("PyPI upload completed successfully.")

    # add a Git tag and push
    git_tag = ctx.run('git tag %s' % version)
    if not git_tag.ok:
        fatal("Failed to add a Git tag for uploaded version %s", version,
              cause=git_tag)
    git_push = ctx.run('git push && git push --tags')
    if not git_push.ok:
        fatal("Failed to push the release upstream.", cause=git_push)
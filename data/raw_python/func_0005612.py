def upload(target):
    # type: (str) -> None
    """ Upload the release to a pypi server.

    TODO: Make sure the git directory is clean before allowing a release.

    Args:
        target (str):
            pypi target as defined in ~/.pypirc
    """
    log.info("Uploading to pypi server <33>{}".format(target))
    with conf.within_proj_dir():
        shell.run('python setup.py sdist register -r "{}"'.format(target))
        shell.run('python setup.py sdist upload -r "{}"'.format(target))
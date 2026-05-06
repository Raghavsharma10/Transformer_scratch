def artifact_already_exists(cli, meta, owner):
    """
    Checks to see whether the built recipe (aka distribution) already
    exists on the owner/user's binstar account.
    """
    distro_name = '{}/{}.tar.bz2'.format(conda.config.subdir, meta.dist())

    try:
        dist_info = cli.distribution(owner, meta.name(), meta.version(), distro_name)
    except binstar_client.errors.NotFound:
        dist_info = {}

    return bool(dist_info)
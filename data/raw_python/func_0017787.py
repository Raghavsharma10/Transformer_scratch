def load_environment(srcdir, datadir=None, allow_old=False):
    """
    Load configuration values for an environment

    :param srcdir: environment source directory
    :param datadir: environment data direcory, if None will be discovered
                    from srcdir
    :param allow_old: Don't throw an exception if this is an old site
                      This is only valid for sites that you are purging.
    if datadir is None it will be discovered from srcdir

    Returns (datadir, name, ckan_version, always_prod, deploy_target,
             remote_server_key)
    """
    cp = ConfigParser.SafeConfigParser()
    try:
        cp.read([srcdir + '/.datacats-environment'])
    except ConfigParser.Error:
        raise DatacatsError('Error reading environment information')

    name = cp.get('datacats', 'name')

    if datadir:
        # update the link in case user moved their srcdir
        save_srcdir_location(datadir, srcdir)
    else:
        datadir = path.expanduser('~/.datacats/' + name)
        # FIXME: check if datadir is sane, project-dir points back to srcdir

    if migrate.needs_format_conversion(datadir) and not allow_old:
        raise DatacatsError('This environment uses an old format. You must'
                            ' migrate to the new format. To do so, use the'
                            ' "datacats migrate" command.')

    if migrate.is_locked(datadir):
        raise DatacatsError('Migration in progress, cannot continue.\n'
                            'If you interrupted a migration, you should'
                            ' attempt manual recovery or contact us by'
                            ' filing an issue at http://github.com/datacats/'
                            'datacats.\nAs a last resort, you could delete'
                            ' all your stored data and create a new environment'
                            ' by running "datacats purge" followed by'
                            ' "datacats init".')

    # FIXME: consider doing data_complete check here

    ckan_version = cp.get('datacats', 'ckan_version')
    try:
        always_prod = cp.getboolean('datacats', 'always_prod')
    except ConfigParser.NoOptionError:
        always_prod = False

    try:
        extra_containers = cp.get('datacats', 'extra_containers').split(' ')
    except ConfigParser.NoOptionError:
        extra_containers = ()

    # if remote_server's custom ssh connection
    # address is defined,
    # we overwrite the default datacats.com one
    try:
        deploy_target = cp.get('deploy', 'remote_server_user') \
            + "@" + cp.get('deploy', 'remote_server')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        deploy_target = DEFAULT_REMOTE_SERVER_TARGET

    # if remote_server's ssh public key is given,
    # we overwrite the default datacats.com one
    try:
        remote_server_key = cp.get('deploy', 'remote_server_key')
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        remote_server_key = None

    return (datadir, name, ckan_version, always_prod, deploy_target,
        remote_server_key, extra_containers)
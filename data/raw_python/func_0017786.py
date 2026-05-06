def find_environment_dirs(environment_name=None, data_only=False):
    """
    :param environment_name: exising environment name, path or None to
        look in current or parent directories for project

    returns (srcdir, extension_dir, datadir)

    extension_dir is the  name of extension directory user was in/referenced,
    default: 'ckan'. This value is used by the paster cli command.

    datadir will be None if environment_name was a path or None (not a name)
    """
    docker.require_images()

    if environment_name is None:
        environment_name = '.'

    extension_dir = 'ckan'
    if validate.valid_name(environment_name) and path.isdir(
            path.expanduser('~/.datacats/' + environment_name)):
        # loading from a name
        datadir = path.expanduser('~/.datacats/' + environment_name)
        with open(datadir + '/project-dir') as pd:
            srcdir = pd.read()

        if not data_only and not path.exists(srcdir + '/.datacats-environment'):
            raise DatacatsError(
                'Environment data found but environment directory is'
                ' missing. Try again from the new environment directory'
                ' location or remove the environment data with'
                ' "datacats purge"')

        return srcdir, extension_dir, datadir

    # loading from a path
    srcdir = path.abspath(environment_name)
    if not path.isdir(srcdir):
        raise DatacatsError('No environment found with that name')

    wd = srcdir
    oldwd = None
    while not path.exists(wd + '/.datacats-environment'):
        oldwd = wd
        wd, _ = path.split(wd)
        if wd == oldwd:
            raise DatacatsError(
                'Environment not found in {0} or above'.format(srcdir))
    srcdir = wd

    if oldwd:
        _, extension_dir = path.split(oldwd)

    return srcdir, extension_dir, None
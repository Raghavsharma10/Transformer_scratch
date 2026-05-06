def new_environment_check(srcpath, site_name, ckan_version):
    """
    Check if a new environment or site can be created at the given path.

    Returns (name, datadir, sitedir, srcdir) or raises DatacatsError
    """
    docker.require_images()

    workdir, name = path.split(path.abspath(path.expanduser(srcpath)))

    if not validate.valid_name(name):
        raise DatacatsError('Please choose an environment name starting'
                            ' with a letter and including only lowercase letters'
                            ' and digits')
    if not path.isdir(workdir):
        raise DatacatsError('Parent directory for environment'
                            ' does not exist')

    datadir = path.expanduser('~/.datacats/' + name)
    sitedir = datadir + '/sites/' + site_name
    # We track through the datadir to the target if we are just making a
    # site
    if path.isdir(datadir):
        with open(datadir + '/project-dir') as pd:
            srcdir = pd.read()
    else:
        srcdir = workdir + '/' + name

    if ckan_version not in SUPPORTED_PRELOADS:
        raise DatacatsError('''Datacats does not currently support CKAN version {}.
Versions that are currently supported are: {}'''.format(ckan_version,
                                                        ', '.join(SUPPORTED_PRELOADS)))

    preload_name = str(ckan_version)

    # Get all the versions from the tags
    downloaded_versions = [tag for tag in docker.get_tags('datacats/ckan')]

    if ckan_version not in downloaded_versions:
        retrying_pull_image('datacats/ckan:{}'.format(preload_name))

    if path.isdir(sitedir):
        raise DatacatsError('Site data directory {0} already exists'.format(
                            sitedir))
    # This is the case where the data dir has been removed,
    if path.isdir(srcdir) and not path.isdir(datadir):
        raise DatacatsError('Environment directory exists, but data directory does not.\n'
                            'If you simply want to recreate the data directory, run '
                            '"datacats init" in the environment directory.')

    return name, datadir, srcdir
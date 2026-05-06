def create_directories(datadir, sitedir, srcdir=None):
    """
    Create expected directories in datadir, sitedir
    and optionally srcdir
    """
    # It's possible that the datadir already exists
    # (we're making a secondary site)
    if not path.isdir(datadir):
        os.makedirs(datadir, mode=0o700)
    try:
        # This should take care if the 'site' subdir if needed
        os.makedirs(sitedir, mode=0o700)
    except OSError:
        raise DatacatsError("Site already exists.")

    # venv isn't site-specific, the rest are.
    if not docker.is_boot2docker():
        if not path.isdir(datadir + '/venv'):
            os.makedirs(datadir + '/venv')
        os.makedirs(sitedir + '/postgres')
    os.makedirs(sitedir + '/solr')
    os.makedirs(sitedir + '/files')
    os.makedirs(sitedir + '/run')

    if srcdir:
        os.makedirs(srcdir)
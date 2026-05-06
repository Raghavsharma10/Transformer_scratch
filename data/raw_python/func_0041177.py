def fetch_pool(repo_url, branch='master', reuse_existing=False):
    """Fetch a git repository from ``repo_url`` and returns a ``FeaturePool`` object."""
    repo_name = get_repo_name(repo_url)
    lib_dir = get_lib_dir()
    pool_dir = get_pool_dir(repo_name)
    print('... fetching %s ' % repo_name)

    if os.path.exists(pool_dir):
        if not reuse_existing:
            raise Exception('ERROR: repository already exists')
    else:
        try:
            a = call(['git', 'clone', repo_url], cwd=lib_dir)
        except OSError:
            raise Exception('ERROR: You probably dont have git installed: sudo apt-get install git')

        if a != 0:
            raise Exception('ERROR: check your repository url and credentials!')

    try:
        call(['git', 'checkout', branch], cwd=pool_dir)
    except OSError:
        raise Exception('ERROR: cannot switch branches')

    print('... repository successfully cloned')
    return FeaturePool(pool_dir)
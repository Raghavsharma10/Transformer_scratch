def git_clone(uri, pull=True, reflect=False, cache_dir=None, chdir=True):
    '''
    Given a git repo, clone (cache) it locally.

    :param uri: git repo uri
    :param pull: whether to pull after cloning (or loading cache)
    '''
    cache_dir = cache_dir or CACHE_DIR
    # make the uri safe for filesystems
    repo_path = os.path.expanduser(os.path.join(cache_dir, safestr(uri)))
    if not os.path.exists(repo_path):
        from_cache = False
        logger.info(
            'Locally caching git repo [%s] to [%s]' % (uri, repo_path))
        cmd = 'git clone %s %s' % (uri, repo_path)
        sys_call(cmd)
    else:
        from_cache = True
        logger.info(
            'GIT repo loaded from local cache [%s])' % (repo_path))
    if pull and not from_cache:
        os.chdir(repo_path)
        cmd = 'git pull'
        sys_call(cmd, cwd=repo_path)
    if chdir:
        os.chdir(repo_path)
    if reflect:
        if not HAS_DULWICH:
            raise RuntimeError("`pip install dulwich` required!")
        return Repo(repo_path)
    else:
        return repo_path
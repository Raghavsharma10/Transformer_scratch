def clone(repo, log, depth=1):
    """Given a list of repositories, make sure they're all cloned.

    Should be called from the subclassed `Catalog` objects, passed a list
    of specific repository names.

    Arguments
    ---------
    all_repos : list of str
        *Absolute* path specification of each target repository.

    """
    kwargs = {}
    if depth > 0:
        kwargs['depth'] = depth

    try:
        repo_name = os.path.split(repo)[-1]
        repo_name = "https://github.com/astrocatalogs/" + repo_name + ".git"
        log.warning("Cloning '{}' (only needs to be done ".format(repo) +
                    "once, may take few minutes per repo).")
        grepo = git.Repo.clone_from(repo_name, repo, **kwargs)
    except:
        log.error("CLONING '{}' INTERRUPTED".format(repo))
        raise

    return grepo
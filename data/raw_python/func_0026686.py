def git_clone_all_repos(cat):
    """Perform a 'git clone' for each data repository that doesnt exist.
    """
    log = cat.log
    log.debug("gitter.git_clone_all_repos()")

    all_repos = cat.PATHS.get_all_repo_folders()
    out_repos = cat.PATHS.get_repo_output_folders()
    for repo in all_repos:
        log.info("Repo in: '{}'".format(repo))

        if os.path.isdir(repo):
            log.info("Directory exists.")
        else:
            log.debug("Cloning directory...")
            clone(repo, cat.log, depth=max(cat.args.clone_depth, 1))

        if cat.args.purge_outputs and repo in out_repos:
            for fil in glob(os.path.join(repo, '*.json')):
                os.remove(fil)

        grepo = git.cmd.Git(repo)
        try:
            grepo.status()
        except git.GitCommandError:
            log.error("Repository does not exist!")
            raise

        # Get the initial git SHA
        sha_beg = get_sha(repo)
        log.debug("Current SHA: '{}'".format(sha_beg))

    return
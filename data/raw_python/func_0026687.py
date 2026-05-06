def git_reset_all_repos(cat, hard=True, origin=False, clean=True):
    """Perform a 'git reset' in each data repository.
    """
    log = cat.log
    log.debug("gitter.git_reset_all_repos()")

    all_repos = cat.PATHS.get_all_repo_folders()
    for repo in all_repos:
        log.warning("Repo in: '{}'".format(repo))
        # Get the initial git SHA
        sha_beg = get_sha(repo)
        log.debug("Current SHA: '{}'".format(sha_beg))

        grepo = git.cmd.Git(repo)
        # Fetch first
        log.info("fetching")
        grepo.fetch()

        args = []
        if hard:
            args.append('--hard')
        if origin:
            args.append('origin/master')
        log.info("resetting")
        retval = grepo.reset(*args)
        if len(retval):
            log.warning("Git says: '{}'".format(retval))

        # Clean
        if clean:
            log.info("cleaning")
            # [q]uiet, [f]orce, [d]irectories
            retval = grepo.clean('-qdf')
            if len(retval):
                log.warning("Git says: '{}'".format(retval))

        sha_end = get_sha(repo)
        if sha_end != sha_beg:
            log.debug("Updated SHA: '{}'".format(sha_end))

    return
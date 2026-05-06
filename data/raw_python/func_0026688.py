def git_status_all_repos(cat, hard=True, origin=False, clean=True):
    """Perform a 'git status' in each data repository.
    """
    log = cat.log
    log.debug("gitter.git_status_all_repos()")

    all_repos = cat.PATHS.get_all_repo_folders()
    for repo_name in all_repos:
        log.info("Repo in: '{}'".format(repo_name))
        # Get the initial git SHA
        sha_beg = get_sha(repo_name)
        log.debug("Current SHA: '{}'".format(sha_beg))

        log.info("Fetching")
        fetch(repo_name, log=cat.log)

        git_comm = ["git", "status"]
        _call_command_in_repo(
            git_comm, repo_name, cat.log, fail=True, log_flag=True)

        sha_end = get_sha(repo_name)
        if sha_end != sha_beg:
            log.info("Updated SHA: '{}'".format(sha_end))

    return
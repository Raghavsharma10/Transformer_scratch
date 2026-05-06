def git_pull_all_repos(cat, strategy_recursive=True, strategy='theirs'):
    """Perform a 'git pull' in each data repository.

    > `git pull -s recursive -X theirs`
    """
    # raise RuntimeError("THIS DOESNT WORK YET!")
    log = cat.log
    log.debug("gitter.git_pull_all_repos()")
    log.warning("WARNING: using experimental `git_pull_all_repos()`!")

    all_repos = cat.PATHS.get_all_repo_folders()
    for repo_name in all_repos:
        log.info("Repo in: '{}'".format(repo_name))
        # Get the initial git SHA
        sha_beg = get_sha(repo_name)
        log.debug("Current SHA: '{}'".format(sha_beg))

        # Initialize the git repository
        repo = git.Repo(repo_name)
        # Construct the command to call
        git_comm = "git pull --verbose"
        if strategy_recursive:
            git_comm += " -s recursive"
        if strategy is not None:
            git_comm += " -X {:s}".format(strategy)
        log.debug("Calling '{}'".format(git_comm))
        # Call git command (do this manually to use desired options)
        #    Set `with_exceptions=False` to handle errors ourselves (below)
        code, out, err = repo.git.execute(
            git_comm.split(),
            with_stdout=True,
            with_extended_output=True,
            with_exceptions=False)
        # Handle output of git command
        if len(out):
            log.info(out)
        if len(err):
            log.info(err)
        # Hangle error-codes
        if code != 0:
            err_str = "Command '{}' returned exit code '{}'!".format(git_comm,
                                                                     code)
            err_str += "\n\tout: '{}'\n\terr: '{}'".format(out, err)
            log.error(err_str)
            raise RuntimeError(err_str)

        sha_end = get_sha(repo_name)
        if sha_end != sha_beg:
            log.info("Updated SHA: '{}'".format(sha_end))

    return
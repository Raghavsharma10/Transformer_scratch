def git_add_commit_push_all_repos(cat):
    """Add all files in each data repository tree, commit, push.

    Creates a commit message based on the current catalog version info.

    If either the `git add` or `git push` commands fail, an error will be
    raised.  Currently, if `commit` fails an error *WILL NOT* be raised
    because the `commit` command will return a nonzero exit status if
    there are no files to add... which we dont want to raise an error.
    FIX: improve the error checking on this.
    """
    log = cat.log
    log.debug("gitter.git_add_commit_push_all_repos()")

    # Do not commit/push private repos
    all_repos = cat.PATHS.get_all_repo_folders(private=False)
    for repo in all_repos:
        log.info("Repo in: '{}'".format(repo))
        # Get the initial git SHA
        sha_beg = get_sha(repo)
        log.debug("Current SHA: '{}'".format(sha_beg))

        # Get files that should be added, compress and check sizes
        add_files = cat._prep_git_add_file_list(repo,
                                                cat.COMPRESS_ABOVE_FILESIZE)
        log.info("Found {} Files to add.".format(len(add_files)))
        if len(add_files) == 0:
            continue

        try:
            # Add all files in the repository directory tree
            git_comm = ["git", "add"]
            if cat.args.travis:
                git_comm.append("-f")
            git_comm.extend(add_files)
            _call_command_in_repo(
                git_comm, repo, cat.log, fail=True, log_flag=False)

            # Commit these files
            commit_msg = "'push' - adding all files."
            commit_msg = "{} : {}".format(cat._version_long, commit_msg)
            log.info(commit_msg)
            git_comm = ["git", "commit", "-am", commit_msg]
            _call_command_in_repo(git_comm, repo, cat.log)

            # Add all files in the repository directory tree
            git_comm = ["git", "push"]
            if not cat.args.travis:
                _call_command_in_repo(git_comm, repo, cat.log, fail=True)
        except Exception as err:
            try:
                git_comm = ["git", "reset", "HEAD"]
                _call_command_in_repo(git_comm, repo, cat.log, fail=True)
            except:
                pass

            raise err

    return
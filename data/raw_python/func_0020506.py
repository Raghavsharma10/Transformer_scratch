def clone_git_repo(git_url, target_dir=None, commit=None, retry_times=GIT_MAX_RETRIES, branch=None,
                   depth=None):
    """
    clone provided git repo to target_dir, optionally checkout provided commit

    :param git_url: str, git repo to clone
    :param target_dir: str, filesystem path where the repo should be cloned
    :param commit: str, commit to checkout, SHA-1 or ref
    :param retry_times: int, number of retries for git clone
    :param branch: str, optional branch of the commit, required if depth is provided
    :param depth: int, optional expected depth
    :return: str, int, commit ID of HEAD
    """
    retry_delay = GIT_BACKOFF_FACTOR
    target_dir = target_dir or os.path.join(tempfile.mkdtemp(), "repo")
    commit = commit or "master"
    logger.info("cloning git repo '%s'", git_url)
    logger.debug("url = '%s', dir = '%s', commit = '%s'",
                 git_url, target_dir, commit)

    cmd = ["git", "clone"]
    if branch:
        cmd += ["-b", branch, "--single-branch"]
        if depth:
            cmd += ["--depth", str(depth)]
    elif depth:
        logger.warning("branch not provided for %s, depth setting ignored", git_url)
        depth = None

    cmd += [git_url, target_dir]

    logger.debug("cloning '%s'", cmd)
    repo_commit = ''
    repo_depth = None
    for counter in range(retry_times + 1):
        try:
            # we are using check_output, even though we aren't using
            # the return value, but we will get 'output' in exception
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            repo_commit, repo_depth = reset_git_repo(target_dir, commit, depth)
            break
        except subprocess.CalledProcessError as exc:
            if counter != retry_times:
                logger.info("retrying command '%s':\n '%s'", cmd, exc.output)
                time.sleep(retry_delay * (2 ** counter))
            else:
                raise OsbsException("Unable to clone git repo '%s' "
                                    "branch '%s'" % (git_url, branch),
                                    cause=exc, traceback=sys.exc_info()[2])

    return ClonedRepoData(target_dir, repo_commit, repo_depth)
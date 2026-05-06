def reset_git_repo(target_dir, git_reference, retry_depth=None):
    """
    hard reset git clone in target_dir to given git_reference

    :param target_dir: str, filesystem path where the repo is cloned
    :param git_reference: str, any valid git reference
    :param retry_depth: int, if the repo was cloned with --shallow, this is the expected
                        depth of the commit
    :return: str and int, commit ID of HEAD and commit depth of git_reference
    """
    deepen = retry_depth or 0
    base_commit_depth = 0
    for _ in range(GIT_FETCH_RETRY):
        try:
            if not deepen:
                cmd = ['git', 'rev-list', '--count', git_reference]
                base_commit_depth = int(subprocess.check_output(cmd, cwd=target_dir)) - 1
            cmd = ["git", "reset", "--hard", git_reference]
            logger.debug("Resetting current HEAD: '%s'", cmd)
            subprocess.check_call(cmd, cwd=target_dir)
            break
        except subprocess.CalledProcessError:
            if not deepen:
                raise OsbsException('cannot find commit %s in repo %s' %
                                    (git_reference, target_dir))
            deepen *= 2
            cmd = ["git", "fetch", "--depth", str(deepen)]
            subprocess.check_call(cmd, cwd=target_dir)
            logger.debug("Couldn't find commit %s, increasing depth with '%s'", git_reference,
                         cmd)
    else:
        raise OsbsException('cannot find commit %s in repo %s' % (git_reference, target_dir))

    cmd = ["git", "rev-parse", "HEAD"]
    logger.debug("getting SHA-1 of provided ref '%s'", git_reference)
    commit_id = subprocess.check_output(cmd, cwd=target_dir, universal_newlines=True)
    commit_id = commit_id.strip()
    logger.info("commit ID = %s", commit_id)

    final_commit_depth = None
    if not deepen:
        cmd = ['git', 'rev-list', '--count', 'HEAD']
        final_commit_depth = int(subprocess.check_output(cmd, cwd=target_dir)) - base_commit_depth

    return commit_id, final_commit_depth
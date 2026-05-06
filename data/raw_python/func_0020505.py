def checkout_git_repo(git_url, target_dir=None, commit=None, retry_times=GIT_MAX_RETRIES,
                      branch=None, depth=None):
    """
    clone provided git repo to target_dir, optionally checkout provided commit
    yield the ClonedRepoData and delete the repo when finished

    :param git_url: str, git repo to clone
    :param target_dir: str, filesystem path where the repo should be cloned
    :param commit: str, commit to checkout, SHA-1 or ref
    :param retry_times: int, number of retries for git clone
    :param branch: str, optional branch of the commit, required if depth is provided
    :param depth: int, optional expected depth
    :return: str, int, commit ID of HEAD
    """
    tmpdir = tempfile.mkdtemp()
    target_dir = target_dir or os.path.join(tmpdir, "repo")
    try:
        yield clone_git_repo(git_url, target_dir, commit, retry_times, branch, depth)
    finally:
        shutil.rmtree(tmpdir)
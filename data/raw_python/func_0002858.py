def update_repo(repo_dir, repo_url, fetch=False):
    """Clone the repo if it doesn't exist already, otherwise update it."""
    repo_exists = os.path.exists(repo_dir)
    if not repo_exists:
        log.info("Cloning repo {}".format(repo_url))
        repo = repo_clone(repo_dir, repo_url)

    # Make sure the repo is properly prepared
    # and has all the refs required
    log.info("Fetching repo {} (fetch: {})".format(repo_url, fetch))
    repo = repo_pull(repo_dir, repo_url, fetch)

    return repo
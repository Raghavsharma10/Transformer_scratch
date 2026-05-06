def validate_commits(repo_dir, commits):
    """Test if a commit is valid for the repository."""
    log.debug("Validating {c} exist in {r}".format(c=commits, r=repo_dir))
    repo = Repo(repo_dir)
    for commit in commits:
        try:
            commit = repo.commit(commit)
        except Exception:
            msg = ("Commit {commit} could not be found in repo {repo}. "
                   "You may need to pass --update to fetch the latest "
                   "updates to the git repositories stored on "
                   "your local computer.".format(repo=repo_dir, commit=commit))
            raise exceptions.InvalidCommitException(msg)

    return True
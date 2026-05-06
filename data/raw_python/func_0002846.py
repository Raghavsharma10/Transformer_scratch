def get_commits(repo_dir, old_commit, new_commit, hide_merges=True):
    """Find all commits between two commit SHAs."""
    repo = Repo(repo_dir)
    commits = repo.iter_commits(rev="{0}..{1}".format(old_commit, new_commit))
    if hide_merges:
        return [x for x in commits if not x.summary.startswith("Merge ")]
    else:
        return list(commits)
def validate_commit_range(repo_dir, old_commit, new_commit):
    """Check if commit range is valid. Flip it if needed."""
    # Are there any commits between the two commits that were provided?
    try:
        commits = get_commits(repo_dir, old_commit, new_commit)
    except Exception:
        commits = []
    if len(commits) == 0:
        # The user might have gotten their commits out of order. Let's flip
        # the order of the commits and try again.
        try:
            commits = get_commits(repo_dir, new_commit, old_commit)
        except Exception:
            commits = []
        if len(commits) == 0:
            # Okay, so there really are no commits between the two commits
            # provided by the user. :)
            msg = ("The commit range {0}..{1} is invalid for {2}."
                   "You may need to use the --update option to fetch the "
                   "latest updates to the git repositories stored on your "
                   "local computer.".format(old_commit, new_commit, repo_dir))
            raise exceptions.InvalidCommitRangeException(msg)
        else:
            return 'flip'

    return True
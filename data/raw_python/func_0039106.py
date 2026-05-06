def upload(branch, user, tool):
    """
    Commit + push to branch
    Returns username, commit hash
    """
    with ProgressBar(_("Uploading")):
        language = os.environ.get("LANGUAGE")
        commit_message = [_("automated commit by {}").format(tool)]

        # If LANGUAGE environment variable is set, we need to communicate
        # this to any remote tool via the commit message.
        if language:
            commit_message.append(f"[{language}]")

        commit_message = " ".join(commit_message)

        # Commit + push
        git = Git(Git.working_area)
        _run(git(f"commit -m {shlex.quote(commit_message)} --allow-empty"))
        _run(git.set(Git.cache)(f"push origin {branch}"))
        commit_hash = _run(git("rev-parse HEAD"))
        return user.name, commit_hash
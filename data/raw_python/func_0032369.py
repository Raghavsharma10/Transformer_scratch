def git_hash():
    """returns the current git hash or unknown if not in git repo"""
    if git_repo() is None:
        return "unknown"
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"])
    # git_hash is a byte string; we want a string.
    git_hash = git_hash.decode("utf-8")
    # git_hash also comes with an extra \n at the end, which we remove.
    git_hash = git_hash.strip()
    return git_hash
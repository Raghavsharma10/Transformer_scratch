def git_pretty():
    """returns a pretty summary of the commit or unkown if not in git repo"""
    if git_repo() is None:
        return "unknown"
    pretty = subprocess.check_output(
        ["git", "log", "--pretty=format:%h %s", "-n", "1"])
    pretty = pretty.decode("utf-8")
    pretty = pretty.strip()
    return pretty
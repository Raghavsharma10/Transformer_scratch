def git_repo():
    """
    Returns the git repository root if the cwd is in a repo, else None
    """
    try:
        reldir = subprocess.check_output(
            ["git", "rev-parse", "--git-dir"])
        reldir = reldir.decode("utf-8")
        return os.path.basename(os.path.dirname(os.path.abspath(reldir)))
    except subprocess.CalledProcessError:
        return None
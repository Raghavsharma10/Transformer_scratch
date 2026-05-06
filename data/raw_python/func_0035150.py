def clone_repo(pkg, dest, repo, repo_dest, branch):
    """Clone the Playdoh repo into a custom path."""
    git(['clone', '--recursive', '-b', branch, repo, repo_dest])
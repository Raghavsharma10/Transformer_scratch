def repo_pull(repo_dir, repo_url, fetch=False):
    """Reset repository and optionally update it."""
    # Make sure the repository is reset to the master branch.
    repo = Repo(repo_dir)
    repo.git.clean("-df")
    repo.git.reset("--hard")
    repo.git.checkout("master")
    repo.head.reset(index=True, working_tree=True)

    # Compile the refspec appropriately to ensure
    # that if the repo is from github it includes
    # all the refs needed, including PR's.
    refspec_list = [
        "+refs/heads/*:refs/remotes/origin/*",
        "+refs/heads/*:refs/heads/*",
        "+refs/tags/*:refs/tags/*"
    ]
    if "github.com" in repo_url:
        refspec_list.extend([
            "+refs/pull/*:refs/remotes/origin/pr/*",
            "+refs/heads/*:refs/remotes/origin/*"])

    # Only get the latest updates if requested.
    if fetch:
        repo.git.fetch(["-u", "-v", "-f",
                        repo_url,
                        refspec_list])
    return repo
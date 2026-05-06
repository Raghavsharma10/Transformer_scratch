def checkout(repo, ref):
    """Checkout a repoself."""
    # Delete local branch if it exists, remote branch will be tracked
    # automatically. This prevents stale local branches from causing problems.
    # It also avoids problems with appending origin/ to refs as that doesn't
    # work with tags, SHAs, and upstreams not called origin.
    if ref in repo.branches:
        # eg delete master but leave origin/master
        log.warn("Removing local branch {b} for repo {r}".format(b=ref,
                                                                 r=repo))
        # Can't delete currently checked out branch, so make sure head is
        # detached before deleting.

        repo.head.reset(index=True, working_tree=True)
        repo.git.checkout(repo.head.commit.hexsha)
        repo.delete_head(ref, '--force')

    log.info("Checkout out repo {repo} to ref {ref}".format(repo=repo,
                                                            ref=ref))
    repo.head.reset(index=True, working_tree=True)
    repo.git.checkout(ref)
    repo.head.reset(index=True, working_tree=True)
    sha = repo.head.commit.hexsha
    log.info("Current SHA for repo {repo} is {sha}".format(repo=repo, sha=sha))
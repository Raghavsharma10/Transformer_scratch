def get_commit_url(repo_url):
    """Determine URL to view commits for repo."""
    if "github.com" in repo_url:
        return repo_url[:-4] if repo_url.endswith(".git") else repo_url
    if "git.openstack.org" in repo_url:
        uri = '/'.join(repo_url.split('/')[-2:])
        return "https://github.com/{0}".format(uri)

    # If it didn't match these conditions, just return it.
    return repo_url
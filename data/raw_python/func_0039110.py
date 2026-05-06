def _get_content(org, repo, branch, filepath):
    """Get all content from org/repo/branch/filepath at GitHub."""
    url = "https://github.com/{}/{}/raw/{}/{}".format(org, repo, branch, filepath)
    r = requests.get(url)
    if not r.ok:
        if r.status_code == 404:
            raise InvalidSlugError(_("Invalid slug. Did you mean to submit something else?"))
        else:
            raise Error(_("Could not connect to GitHub."))
    return r.content
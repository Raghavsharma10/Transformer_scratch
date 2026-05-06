def parse_editable(editable_req, default_vcs=None):
    """Parses svn+http://blahblah@rev#egg=Foobar into a requirement
    (Foobar) and a URL"""
    url = editable_req
    if os.path.isdir(url) and os.path.exists(os.path.join(url, 'setup.py')):
        # Treating it as code that has already been checked out
        url = path_to_url(url)
    if url.lower().startswith('file:'):
        return None, url
    for version_control in vcs:
        if url.lower().startswith('%s:' % version_control):
            url = '%s+%s' % (version_control, url)
    if '+' not in url:
        if default_vcs:
            url = default_vcs + '+' + url
        else:
            raise InstallationError(
                '--editable=%s should be formatted with svn+URL, git+URL, hg+URL or bzr+URL' % editable_req)
    vc_type = url.split('+', 1)[0].lower()
    if not vcs.get_backend(vc_type):
        raise InstallationError(
            'For --editable=%s only svn (svn+URL), Git (git+URL), Mercurial (hg+URL) and Bazaar (bzr+URL) is currently supported' % editable_req)
    match = re.search(r'(?:#|#.*?&)egg=([^&]*)', editable_req)
    if (not match or not match.group(1)) and vcs.get_backend(vc_type):
        parts = [p for p in editable_req.split('#', 1)[0].split('/') if p]
        if parts[-2] in ('tags', 'branches', 'tag', 'branch'):
            req = parts[-3]
        elif parts[-1] == 'trunk':
            req = parts[-2]
        else:
            raise InstallationError(
                '--editable=%s is not the right format; it must have #egg=Package'
                % editable_req)
    else:
        req = match.group(1)
    ## FIXME: use package_to_requirement?
    match = re.search(r'^(.*?)(?:-dev|-\d.*)$', req)
    if match:
        # Strip off -dev, -0.2, etc.
        req = match.group(1)
    return req, url
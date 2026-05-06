def _slug_from_url(url):
    """ Parses a project slug out of either an HTTPS or SSH URL. """
    http_match = _HTTPS_REGEX.match(url)
    ssh_match = _SSH_REGEX.match(url)
    if not http_match and not ssh_match:
        raise RuntimeError('Could not parse the URL (`%s`) '
                           'for your repository.' % url)
    if http_match:
        return '/'.join(http_match.groups())
    else:
        return '/'.join(ssh_match.groups())
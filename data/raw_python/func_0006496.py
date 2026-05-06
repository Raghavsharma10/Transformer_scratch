def enable_apt_repositories(prefix, url, version, repositories):
    """ adds an apt repository """
    with settings(hide('warnings', 'running', 'stdout'),
                  warn_only=False, capture=True):
        sudo('apt-add-repository "%s %s %s %s"' % (prefix,
                                                   url,
                                                   version,
                                                   repositories))
        with hide('running', 'stdout'):
            output = sudo("DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get update")
        if 'Some index files failed to download' in output:
            raise SystemExit(1)
        else:
            # if we didn't abort above, we should return True
            return True
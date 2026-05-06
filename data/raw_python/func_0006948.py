def is_deb_package_installed(pkg):
    """ checks if a particular deb package is installed """

    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=True, capture=True):

        result = sudo('dpkg-query -l "%s" | grep -q ^.i' % pkg)
        return not bool(result.return_code)
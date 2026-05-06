def yum_install_from_url(pkg_name, url):
    """ installs a pkg from a url
        p pkg_name: the name of the package to install
        p url: the full URL for the rpm package
    """
    if is_package_installed(distribution='el', pkg=pkg_name) is False:
        log_green(
            "installing %s from %s" % (pkg_name, url))
        with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                      warn_only=True, capture=True):

            result = sudo("rpm -i %s" % url)
            if result.return_code == 0:
                return True
            elif result.return_code == 1:
                return False
            else:  # print error to user
                print(result)
                raise SystemExit()
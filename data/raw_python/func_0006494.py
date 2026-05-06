def apt_install_from_url(pkg_name, url, log=False):
    """ installs a pkg from a url
        p pkg_name: the name of the package to install
        p url: the full URL for the rpm package
    """
    if is_package_installed(distribution='ubuntu', pkg=pkg_name) is False:

        if log:
            log_green(
                "installing %s from %s" % (pkg_name, url))

        with settings(hide('warnings', 'running', 'stdout'),
                      capture=True):

            sudo("wget -c -O %s.deb %s" % (pkg_name, url))
            sudo("dpkg -i %s.deb" % pkg_name)
            # if we didn't abort above, we should return True
            return True
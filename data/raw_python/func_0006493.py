def apt_install(**kwargs):
    """
        installs a apt package
    """
    for pkg in list(kwargs['packages']):
        if is_package_installed(distribution='ubuntu', pkg=pkg) is False:
            sudo("DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install -y %s" % pkg)
        # if we didn't abort above, we should return True
        return True
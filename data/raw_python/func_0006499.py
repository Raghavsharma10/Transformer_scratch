def is_package_installed(distribution, pkg):
    """ checks if a particular package is installed """
    if ('centos' in distribution or
            'el' in distribution or
            'redhat' in distribution):
        return(is_rpm_package_installed(pkg))

    if ('ubuntu' in distribution or
            'debian' in distribution):
        return(is_deb_package_installed(pkg))
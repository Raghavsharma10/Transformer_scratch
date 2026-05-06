def install_os_updates(distribution, force=False):
    """ installs OS updates """
    if ('centos' in distribution or
            'rhel' in distribution or
            'redhat' in distribution):
        bookshelf2.logging_helpers.log_green('installing OS updates')
        sudo("yum -y --quiet clean all")
        sudo("yum group mark convert")
        sudo("yum -y --quiet update")

    if ('ubuntu' in distribution or
            'debian' in distribution):
        with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                      warn_only=False, capture=True):
            sudo("DEBIAN_FRONTEND=noninteractive apt-get update")
            if force:
                sudo("sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o "
                     "Dpkg::Options::='--force-confdef' "
                     "-o Dpkg::Options::='--force-confold' upgrade --force-yes")
            else:
                sudo("sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o "
                     "Dpkg::Options::='--force-confdef' -o "
                     "Dpkg::Options::='--force-confold' upgrade")
def add_zfs_apt_repository():
    """ adds the ZFS repository """
    with settings(hide('warnings', 'running', 'stdout'),
                  warn_only=False, capture=True):
        sudo('DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get update')
        install_ubuntu_development_tools()
        apt_install(packages=['software-properties-common',
                              'dkms',
                              'linux-headers-generic',
                              'build-essential'])
        sudo('echo | add-apt-repository ppa:zfs-native/stable')
        sudo('DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get update')
        return True
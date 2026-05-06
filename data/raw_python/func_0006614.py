def install_virtualbox(distribution, force_setup=False):
    """ install virtualbox """

    if 'ubuntu' in distribution:
        with hide('running', 'stdout'):
            sudo('DEBIAN_FRONTEND=noninteractive apt-get update')
            sudo("sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o "
                    "Dpkg::Options::='--force-confdef' "
                    "-o Dpkg::Options::='--force-confold' upgrade --force-yes")
            install_ubuntu_development_tools()
            apt_install(packages=['dkms',
                                  'linux-headers-generic',
                                  'build-essential'])
            sudo('wget -q '
                 'https://www.virtualbox.org/download/oracle_vbox.asc -O- |'
                 'sudo apt-key add -')

            os = lsb_release()
            apt_string = ' '.join(
                ['deb',
                 'http://download.virtualbox.org/virtualbox/debian',
                 '%s contrib' % os['DISTRIB_CODENAME']])

            apt_add_repository_from_apt_string(apt_string, 'vbox.list')

            apt_install(packages=['virtualbox-5.0'])

            loaded_modules = sudo('lsmod')

            if 'vboxdrv' not in loaded_modules or force_setup:

                if 'Vivid Vervet' in run('cat /etc/os-release'):
                    sudo('systemctl start vboxdrv')
                else:
                    sudo('/etc/init.d/vboxdrv start')

            sudo('wget -c '
                 'http://download.virtualbox.org/virtualbox/5.0.4/'
                 'Oracle_VM_VirtualBox_Extension_Pack-5.0.4-102546.vbox-extpack') # noqa

            sudo('VBoxManage extpack install --replace '
                 'Oracle_VM_VirtualBox_Extension_Pack-5.0.4-102546.vbox-extpack')
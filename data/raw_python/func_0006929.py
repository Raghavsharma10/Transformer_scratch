def disable_selinux():
    """ disables selinux """

    if contains(filename='/etc/selinux/config',
                text='SELINUX=enforcing'):
        sed('/etc/selinux/config',
            'SELINUX=enforcing', 'SELINUX=disabled', use_sudo=True)

    if contains(filename='/etc/selinux/config',
                text='SELINUX=permissive'):
        sed('/etc/selinux/config',
            'SELINUX=permissive', 'SELINUX=disabled', use_sudo=True)

    if sudo('getenforce').lower() != 'disabled':
        with settings(warn_only=True, capture=True):
            sudo('/sbin/reboot')
        sleep_for_one_minute()
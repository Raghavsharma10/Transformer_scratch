def enable_marathon_basic_authentication(principal, password):
    """ configures marathon to start with authentication """
    upstart_file = '/etc/init/marathon.conf'
    with hide('running', 'stdout'):
        sudo('echo -n "{}" > /etc/marathon-mesos.credentials'.format(password))
    boot_args = ' '.join(['exec',
                          '/usr/bin/marathon',
                          '--http_credentials',
                          '"{}:{}"'.format(principal, password),
                          '--mesos_authentication_principal',
                          principal,
                          '--mesos_authentication_secret_file',
                          '/etc/marathon-mesos.credentials'])

    # check if the init conf file contains the exact user and password
    if not file_contains(upstart_file, boot_args, use_sudo=True):
        sed(upstart_file, 'exec /usr/bin/marathon.*', boot_args, use_sudo=True)
        file_attribs(upstart_file, mode=700, sudo=True)
        restart_service('marathon')
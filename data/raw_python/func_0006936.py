def enable_mesos_basic_authentication(principal, password):
    """ enables and adds a new authorized principal """
    restart = False
    secrets_file = '/etc/mesos/secrets'
    secrets_entry = '%s %s' % (principal, password)
    if not file_contains(filename=secrets_file,
                         text=secrets_entry, use_sudo=True):
        file_append(filename=secrets_file, text=secrets_entry, use_sudo=True)
        file_attribs(secrets_file, mode=700, sudo=True)
        restart = True

    # set new startup parameters for mesos-master
    with quiet():
        if secrets_file not in sudo('cat /etc/mesos-master/credentials'):
            sudo('echo %s > /etc/mesos-master/credentials' % secrets_file)
            restart = True

        if not exists('/etc/mesos-master/\?authenticate', use_sudo=True):
            sudo('touch /etc/mesos-master/\?authenticate')
            file_attribs('/etc/mesos-master/\?authenticate',
                         mode=700,
                         sudo=True)
            restart = True

    if restart:
        restart_service('mesos-master')
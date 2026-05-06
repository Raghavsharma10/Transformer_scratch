def install_mesos_single_box_mode(distribution):
    """ install mesos (all of it) on a single node"""

    if 'ubuntu' in distribution:
        log_green('adding mesosphere apt-key')
        apt_add_key(keyid='E56151BF')

        os = lsb_release()
        apt_string = 'deb http://repos.mesosphere.io/%s %s main' % (
            os['DISTRIB_ID'], os['DISTRIB_CODENAME'])

        log_green('adding mesosphere apt repository')
        apt_add_repository_from_apt_string(apt_string, 'mesosphere.list')

        log_green('installing ubuntu development tools')
        install_ubuntu_development_tools()

        install_oracle_java(distribution, '8')

        log_green('installing mesos and marathon')
        apt_install(packages=['mesos', 'marathon'])

        if not file_contains('/etc/default/mesos-master',
                             'MESOS_QUORUM=1', use_sudo=True):
            file_append('/etc/default/mesos-master',
                        'MESOS_QUORUM=1', use_sudo=True)

            log_green('restarting services...')
            for svc in ['zookeeper', 'mesos-master', 'mesos-slave', 'marathon']:
                restart_service(svc)

        if not file_contains('/etc/mesos-slave/work_dir',
                             '/data/mesos', use_sudo=True):
            file_append('/etc/mesos-slave/work_dir',
                        '/data/mesos', use_sudo=True)

            log_green('restarting services...')
            for svc in ['mesos-slave']:
                restart_service(svc)

        log_green('enabling nginx autoindex on /...')

        with quiet():
            cmd = 'cat /etc/nginx/sites-available/default'
            contents = sudo(cmd).replace('\n', ' ').replace('\r', '')

        if not bool(re.search('.*#*location \/ {.*autoindex on;.*', contents)):
            insert_line_in_file_after_regex(
                path='/etc/nginx/sites-available/default',
                line='                autoindex on;',
                after_regex='^[^#]*location \/ {',
                use_sudo=True)
            log_green('restarting nginx')
            restart_service('nginx')
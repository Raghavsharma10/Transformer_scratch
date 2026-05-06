def add_firewalld_service(service, permanent=True):
    """ adds a firewall rule """
    yum_install(packages=['firewalld'])

    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=True, capture=True):
        p = ''
        if permanent:
            p = '--permanent'
        sudo('firewall-cmd --add-service %s %s' % (service, p))
        sudo('systemctl reload firewalld')
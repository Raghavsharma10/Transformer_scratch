def systemd(service, start=True, enabled=True, unmask=False, restart=False):
    """ manipulates systemd services """

    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=True, capture=True):

        if restart:
            sudo('systemctl restart %s' % service)
        else:
            if start:
                sudo('systemctl start %s' % service)
            else:
                sudo('systemctl stop %s' % service)

        if enabled:
            sudo('systemctl enable %s' % service)
        else:
            sudo('systemctl disable %s' % service)

        if unmask:
            sudo('systemctl unmask %s' % service)
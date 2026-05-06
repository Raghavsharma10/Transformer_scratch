def restart_service(service):
    """ restarts a service  """
    with settings(hide('running', 'stdout'), warn_only=True):
        log_yellow('stoping service %s' % service)
        sudo('service %s stop' % service)
        log_yellow('starting service %s' % service)
        sudo('service %s start' % service)
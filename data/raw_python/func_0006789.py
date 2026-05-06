def restart_service(service, log=False):
    """ restarts a service  """
    with settings():
        if log:
            bookshelf2.logging_helpers.log_yellow(
                'stoping service %s' % service)
        sudo('service %s stop' % service)
        if log:
            bookshelf2.logging_helpers.log_yellow(
                'starting service %s' % service)
        sudo('service %s start' % service)
    return True
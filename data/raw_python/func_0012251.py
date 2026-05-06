def connection(model_connections):
    """
        Creates the example directory structure necessary for a connection
        service.
    """

    # for each connection group
    for connection_str in model_connections:

        # the services to connect
        services = connection_str.split(':')
        services.sort()

        service_name = ''.join([service.title() for service in services])

        # the template context
        context = {
            # make sure the first letter is lowercase
            'name': service_name[0].lower() + service_name[1:],
            'services': services,
        }

        render_template(template='common', context=context)
        render_template(template='connection', context=context)
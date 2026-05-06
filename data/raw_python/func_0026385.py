def show(ctx, component):
    """Show the stored, active configuration of a component."""

    col = ctx.obj['col']

    if col.count({'name': component}) > 1:
        log('More than one component configuration of this name! Try '
            'one of the uuids as argument. Get a list with "config '
            'list"')
        return

    if component is None:
        configurations = col.find()
        for configuration in configurations:
            log("%-15s : %s" % (configuration.name,
                                configuration.uuid),
                emitter='MANAGE')
    else:
        configuration = col.find_one({'name': component})

        if configuration is None:
            configuration = col.find_one({'uuid': component})

        if configuration is None:
            log('No component with that name or uuid found.')
            return

        print(json.dumps(configuration.serializablefields(), indent=4))
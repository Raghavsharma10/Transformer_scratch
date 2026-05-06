def delete(ctx, componentname):
    """Delete an existing component configuration. This will trigger
    the creation of its default configuration upon next restart."""
    col = ctx.obj['col']

    if col.count({'name': componentname}) > 1:
        log('More than one component configuration of this name! Try '
            'one of the uuids as argument. Get a list with "config '
            'list"')
        return

    log('Deleting component configuration', componentname,
        emitter='MANAGE')

    configuration = col.find_one({'name': componentname})

    if configuration is None:
        configuration = col.find_one({'uuid': componentname})

    if configuration is None:
        log('Component configuration not found:', componentname,
            emitter='MANAGE')
        return

    configuration.delete()
    log('Done')
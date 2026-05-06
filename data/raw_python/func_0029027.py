def load_plugins(group='gcdt10'):
    """Load and register installed gcdt plugins.
    """
    # on using entrypoints:
    # http://stackoverflow.com/questions/774824/explain-python-entry-points
    # TODO: make sure we do not have conflicting generators installed!
    for ep in pkg_resources.iter_entry_points(group, name=None):
        plugin = ep.load()  # load the plugin
        if check_hook_mechanism_is_intact(plugin):
            if check_register_present(plugin):
                plugin.register()   # register the plugin so it listens to gcdt_signals
        else:
            log.warning('No valid hook configuration: %s. Not using hooks!', plugin)
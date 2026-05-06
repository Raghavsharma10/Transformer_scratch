def __discoverPlugins():
    """ Discover the plugin classes contained in Python files, given a
        list of directory names to scan. Return a list of plugin classes.
    """
    for app in settings.INSTALLED_APPS:
        if not app.startswith('django'):
            module = __import__(app)
            moduledir = path.Path(module.__file__).parent
            plugin = moduledir / 'frog_plugin.py'
            if plugin.exists():
                file_, fpath, desc = imp.find_module('frog_plugin', [moduledir])
                if file_:
                    imp.load_module('frog_plugin', file_, fpath, desc)

    return FrogPluginRegistry.plugins
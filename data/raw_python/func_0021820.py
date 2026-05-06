def load_plugins(builtin=True, others=True):
    """Load plugins, either builtin, others, or both.
    """
    for entry_point in pkg_resources.iter_entry_points('yolk.plugins'):
        #LOG.debug("load plugin %s" % entry_point)
        try:
            plugin = entry_point.load()
        except KeyboardInterrupt:
            raise
        except Exception as err_msg:
            # never want a plugin load to exit yolk
            # but we can't log here because the logger is not yet
            # configured
            warn("Unable to load plugin %s: %s" % \
                    (entry_point, err_msg), RuntimeWarning)
            continue
        if plugin.__module__.startswith('yolk.plugins'):
            if builtin:
                yield plugin
        elif others:
            yield plugin
def run(conf, only):
    """Runs uWSGI passing to it using the default or another `uwsgiconf` configuration module.

    """
    with errorprint():
        config = ConfModule(conf)
        spawned = config.spawn_uwsgi(only)

        for alias, pid in spawned:
            click.secho("Spawned uWSGI for configuration aliased '%s'. PID %s" % (alias, pid), fg='green')
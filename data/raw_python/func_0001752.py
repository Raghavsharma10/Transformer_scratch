def _services(lancet):
    """List all currently configured services."""
    def get_services(config):
        for s in config.sections():
            if config.has_option(s, 'url'):
                if config.has_option(s, 'username'):
                    yield s

    for s in get_services(lancet.config):
        click.echo('{}[Logout from {}]'.format(s, lancet.config.get(s, 'url')))
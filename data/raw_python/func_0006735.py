def includeme(config):
    """this function adds some configuration for the application"""
    config.add_route('references', '/references')
    _add_referencer(config.registry)
    config.add_view_deriver(protected_resources.protected_view)
    config.add_renderer('json_item', json_renderer)
    config.scan()
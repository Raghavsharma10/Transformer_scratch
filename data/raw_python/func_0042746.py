def register_views(*args):
    """ Registration view for each resource from config.
    """
    config = args[0]
    settings = config.get_settings()
    pages_config = settings[CONFIG_MODELS]
    resources = resources_of_config(pages_config)
    for resource in resources:
        if hasattr(resource, '__table__')\
                and not hasattr(resource, 'model'):
            continue
        resource.model.pyramid_pages_template = resource.template
        config.add_view(resource.view,
                        attr=resource.attr,
                        route_name=PREFIX_PAGE,
                        renderer=resource.template,
                        context=resource,
                        permission=PREFIX_PAGE)
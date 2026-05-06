def page_factory(request):
    """ Page factory.

    Config models example:

    .. code-block:: python

        models = {
            '': [WebPage, CatalogResource],
            'catalogue': CatalogResource,
            'news': NewsResource,
        }
    """
    prefix = request.matchdict['prefix']  # /{prefix}/page1/page2/page3...
    settings = request.registry.settings
    dbsession = settings[CONFIG_DBSESSION]
    config = settings[CONFIG_MODELS]

    if prefix not in config:
        # prepend {prefix} to *traverse
        request.matchdict['traverse'] =\
            tuple([prefix] + list(request.matchdict['traverse']))
        prefix = None

    # Get all resources and models from config with the same prefix.
    resources = config.get(
        prefix, config.get(   # 1. get resources with prefix same as URL prefix
            '', config.get(   # 2. if not, then try to get empty prefix
                '/', None)))  # 3. else try to get prefix '/' otherwise None

    if not hasattr(resources, '__iter__'):
        resources = (resources, )

    tree = {}

    if not resources:
        return tree

    # Add top level nodes of resources in the tree
    for resource in resources:
        table = None
        if not hasattr(resource, '__table__')\
                and hasattr(resource, 'model'):
            table = resource.model
        else:
            table = resource

        if not hasattr(table, 'slug'):
            continue

        nodes = dbsession.query(table)
        if hasattr(table, 'parent_id'):
            nodes = nodes.filter(or_(
                table.parent_id == None,  # noqa
                table.parent.has(table.slug == '/')
            ))
        for node in nodes:
            if not node.slug:
                continue
            resource = resource_of_node(resources, node)
            tree[node.slug] = resource(node, prefix=prefix)
    return tree
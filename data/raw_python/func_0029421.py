def add_resource_routes(config, view, member_name, collection_name, **kwargs):
    """
    ``view`` is a dotted name of (or direct reference to) a
    Python view class,
    e.g. ``'my.package.views.MyView'``.

    ``member_name`` should be the appropriate singular version of the resource
    given your locale and used with members of the collection.

    ``collection_name`` will be used to refer to the resource collection
    methods and should be a plural version of the member_name argument.

    All keyword arguments are optional.

    ``path_prefix``
        Prepends the URL path for the Route with the path_prefix
        given. This is most useful for cases where you want to mix
        resources or relations between resources.

    ``name_prefix``
        Prepends the route names that are generated with the
        name_prefix given. Combined with the path_prefix option,
        it's easy to generate route names and paths that represent
        resources that are in relations.

        Example::

            config.add_resource_routes(
                'myproject.views:CategoryView', 'message', 'messages',
                path_prefix='/category/{category_id}',
                name_prefix="category_")

            # GET /category/7/messages/1
            # has named route "category_message"

    """

    view = maybe_dotted(view)
    path_prefix = kwargs.pop('path_prefix', '')
    name_prefix = kwargs.pop('name_prefix', '')

    if config.route_prefix:
        name_prefix = "%s_%s" % (config.route_prefix, name_prefix)

    if collection_name:
        id_name = '/{%s}' % (kwargs.pop('id_name', None) or DEFAULT_ID_NAME)
    else:
        id_name = ''

    path = path_prefix.strip('/') + '/' + (collection_name or member_name)

    _factory = kwargs.pop('factory', None)
    # If factory is not set, than auth should be False
    _auth = kwargs.pop('auth', None) and _factory
    _traverse = (kwargs.pop('traverse', None) or id_name) if _factory else None

    action_route = {}
    added_routes = {}

    def add_route_and_view(config, action, route_name, path, request_method,
                           **route_kwargs):
        if route_name not in added_routes:
            config.add_route(
                route_name, path, factory=_factory,
                request_method=['GET', 'POST', 'PUT', 'PATCH', 'DELETE',
                                'OPTIONS'],
                **route_kwargs)
            added_routes[route_name] = path

        action_route[action] = route_name

        if _auth:
            permission = PERMISSIONS[action]
        else:
            permission = None
        config.add_view(view=view, attr=action, route_name=route_name,
                        request_method=request_method,
                        permission=permission,
                        **kwargs)
        config.commit()

    if collection_name == member_name:
        collection_name = collection_name + '_collection'

    if collection_name:
        add_route_and_view(
            config, 'index', name_prefix + collection_name, path,
            'GET')

        add_route_and_view(
            config, 'collection_options', name_prefix + collection_name, path,
            'OPTIONS')

    add_route_and_view(
        config, 'show', name_prefix + member_name, path + id_name,
        'GET', traverse=_traverse)

    add_route_and_view(
        config, 'item_options', name_prefix + member_name, path + id_name,
        'OPTIONS', traverse=_traverse)

    add_route_and_view(
        config, 'replace', name_prefix + member_name, path + id_name,
        'PUT', traverse=_traverse)

    add_route_and_view(
        config, 'update', name_prefix + member_name, path + id_name,
        'PATCH', traverse=_traverse)

    add_route_and_view(
        config, 'create', name_prefix + (collection_name or member_name), path,
        'POST')

    add_route_and_view(
        config, 'delete', name_prefix + member_name, path + id_name,
        'DELETE', traverse=_traverse)

    if collection_name:
        add_route_and_view(
            config, 'update_many',
            name_prefix + (collection_name or member_name),
            path, 'PUT', traverse=_traverse)

        add_route_and_view(
            config, 'update_many',
            name_prefix + (collection_name or member_name),
            path, 'PATCH', traverse=_traverse)

        add_route_and_view(
            config, 'delete_many',
            name_prefix + (collection_name or member_name),
            path, 'DELETE', traverse=_traverse)

    return action_route
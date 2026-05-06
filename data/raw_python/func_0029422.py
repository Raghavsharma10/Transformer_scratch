def get_default_view_path(resource):
    "Returns the dotted path to the default view class."

    parts = [a.member_name for a in resource.ancestors] +\
            [resource.collection_name or resource.member_name]

    if resource.prefix:
        parts.insert(-1, resource.prefix)

    view_file = '%s' % '_'.join(parts)
    view = '%s:%sView' % (view_file, snake2camel(view_file))

    app_package_name = get_app_package_name(resource.config)
    return '%s.views.%s' % (app_package_name, view)
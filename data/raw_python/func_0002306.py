def group_plugins_into_categories(plugins):
    """
    Return all plugins, grouped by category.
    The structure is a {"Categorynane": [list of plugin classes]}
    """
    if not plugins:
        return {}
    plugins = sorted(plugins, key=lambda p: p.verbose_name)
    categories = {}

    for plugin in plugins:
        title = str(plugin.category or u"")  # enforce resolving ugettext_lazy proxies.
        if title not in categories:
            categories[title] = []
        categories[title].append(plugin)

    return categories
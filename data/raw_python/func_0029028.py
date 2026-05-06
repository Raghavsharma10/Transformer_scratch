def get_plugin_versions(group='gcdt10'):
    """Load and register installed gcdt plugins.
    """
    versions = {}
    for ep in pkg_resources.iter_entry_points(group, name=None):
        versions[ep.dist.project_name] = ep.dist.version

    return versions
def print_plugin_list(plugins: Dict[str, pkg_resources.EntryPoint]):
    """
    Prints all registered plugins and checks if they can be loaded or not.

    :param plugins: plugins
    :type plugins: Dict[str, ~pkg_resources.EntryPoint]
    """
    for trigger, entry_point in plugins.items():
        try:
            plugin_class = entry_point.load()
            version = str(plugin_class._info.version)
            print(
                f"{trigger} (ok)\n"
                f"    {version}"
            )
        except Exception:
            print(
                f"{trigger} (failed)"
            )
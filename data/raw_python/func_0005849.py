def probe_plugins():
    """Runs uWSGI to determine what plugins are available and prints them out.

    Generic plugins come first then after blank line follow request plugins.

    """
    plugins = UwsgiRunner().get_plugins()

    for plugin in sorted(plugins.generic):
        click.secho(plugin)

    click.secho('')

    for plugin in sorted(plugins.request):
        click.secho(plugin)
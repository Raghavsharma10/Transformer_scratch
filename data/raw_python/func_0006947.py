def is_vagrant_plugin_installed(plugin, use_sudo=False):
    """ checks if vagrant plugin is installed """

    cmd = 'vagrant plugin list'

    if use_sudo:
        results = sudo(cmd)
    else:
        results = run(cmd)

    installed_plugins = []
    for line in results:
        plugin = re.search('^(\S.*) \((.*)\)$', line)
        installed_plugins.append({'name': plugin.group(0),
                                  'version': plugin.group(1)})
        return installed_plugins
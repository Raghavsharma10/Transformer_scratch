def parse_command_plugins_output(out):
    """Parses ``plugin-list`` command output from uWSGI
    and returns object containing lists of embedded plugin names.

    :param str|unicode out:

    :rtype EmbeddedPlugins:

    """
    out = out.split('--- end of plugins list ---')[0]
    out = out.partition('plugins ***')[2]
    out = out.splitlines()

    current_slot = 0

    plugins = EmbeddedPlugins([], [])

    for line in out:
        line = line.strip()

        if not line:
            continue

        if line.startswith('***'):
            current_slot += 1
            continue

        if current_slot is not None:
            plugins[current_slot].append(line)

    plugins = plugins._replace(request=[plugin.partition(' ')[2] for plugin in plugins.request])

    return plugins
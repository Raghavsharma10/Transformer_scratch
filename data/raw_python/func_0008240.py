def run(plugin_name: str, options: List[str] = None) -> PluginState:
    """
    Run a plugin so use the download routine and clean up after.

    :param plugin_name: name of plugin
    :type plugin_name: str
    :param options: parameters which will be send to the plugin initialization
    :type options: List[str]
    :return: success
    :rtype: ~unidown.plugin.plugin_state.PluginState
    """
    if options is None:
        options = []

    if plugin_name not in dynamic_data.AVAIL_PLUGINS:
        msg = 'Plugin ' + plugin_name + ' was not found.'
        logging.error(msg)
        print(msg)
        return PluginState.NOT_FOUND

    try:
        plugin_class = dynamic_data.AVAIL_PLUGINS[plugin_name].load()
        plugin = plugin_class(options)
    except Exception:
        msg = 'Plugin ' + plugin_name + ' crashed while loading.'
        logging.exception(msg)
        print(msg + ' Check log for more information.')
        return PluginState.LOAD_CRASH
    else:
        logging.info('Loaded plugin: ' + plugin_name)

    try:
        download_from_plugin(plugin)
        plugin.clean_up()
    except PluginException as ex:
        msg = f"Plugin {plugin.name} stopped working. Reason: {'unknown' if (ex.msg == '') else ex.msg}"
        logging.error(msg)
        print(msg)
        return PluginState.RUN_FAIL
    except Exception:
        msg = 'Plugin ' + plugin.name + ' crashed.'
        logging.exception(msg)
        print(msg + ' Check log for more information.')
        return PluginState.RUN_CRASH
    else:
        logging.info(plugin.name + ' ends without errors.')
        return PluginState.END_SUCCESS
def set_plugins_params(self, plugins=None, search_dirs=None, autoload=None, required=False):
        """Sets plugin-related parameters.

        :param list|str|unicode|OptionsGroup|list[OptionsGroup] plugins: uWSGI plugins to load

        :param list|str|unicode search_dirs: Directories to search for uWSGI plugins.

        :param bool autoload: Try to automatically load plugins when unknown options are found.

        :param bool required: Load uWSGI plugins and exit on error.

        """
        plugins = plugins or []

        command = 'need-plugin' if required else 'plugin'

        for plugin in listify(plugins):

            if plugin not in self._plugins:
                self._set(command, plugin, multi=True)
                self._plugins.append(plugin)

        self._set('plugins-dir', search_dirs, multi=True, priority=0)
        self._set('autoload', autoload, cast=bool)

        return self
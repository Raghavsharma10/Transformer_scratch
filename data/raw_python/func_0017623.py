def reload(self):
        """
        Reload the installed plugins and the configuration file. This is called
        when either the plugins or config get updated.
        """
        logging.debug('Reloading config file as files have been modified.')
        self.config.plugin, self.config.general = self.config.reload()
        logging.debug('Reloading plugins as files have been modified.')
        self.loader = plugin_manager.PluginLoader(
            self._plugin_path, self.config.plugin)
        self._plugin_mod_time = os.path.getmtime(self._plugin_path)
        self._config_mod_time = os.path.getmtime(self._config_file_path)
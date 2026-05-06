def load_objects(self):
        """
        Matches the plugins that have been specified in the config file
        with the available plugins. Returns instantiated objects based upon
        the classes defined in the plugins.
        """
        objects = []
        for settings in self._config:
            if settings['plugin'] in self.plugins:
                module = self.plugins[settings['plugin']]
                # Trusts that the only item in __all__ is the name of the
                # plugin class.
                plugin_class = getattr(module, module.__all__)
                objects.append(plugin_class(settings))
                logging.debug('Loaded a plugin object based upon {0}'.format(
                    settings['plugin']))
            else:
                logging.critical('Missing plugin {0} was not found in {1}'.format(
                    settings['plugin'], self.dir_path))
                raise MissingPlugin('The plugin {0} was not found in {1}'.format(
                    settings['plugin'], self.dir_path))
        return objects
def reload(self):
        """
        Reload the configuration from the file. This is in its own function
        so that it can be called at any time by another class.
        """
        self._conf = configparser.ConfigParser()
        # Preserve the case of sections and keys.
        self._conf.optionxform = str
        self._conf.read(self.config_file_path)
        if 'general' not in self._conf.keys():
            raise IncompleteConfigurationFile('Missing the general section')
        general = self._replace_data_types(dict(self._conf.items('general')))
        self._conf.remove_section('general')
        plugin = []
        for section in self._conf.sections():
            plugin.append(dict(self._conf.items(section)))
            plugin[-1].update({'name': section})
            plugin[-1] = self._replace_data_types(plugin[-1])
        return (plugin, general)
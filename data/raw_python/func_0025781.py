def _loadConfig(self):
        """
        loads configuration from some predictable locations.
        :return: the config.
        """
        configPath = path.join(self._getConfigPath(), self._name + ".yml")
        if os.path.exists(configPath):
            self.logger.warning("Loading config from " + configPath)
            with open(configPath, 'r') as yml:
                return yaml.load(yml, Loader=yaml.FullLoader)
        defaultConfig = self.loadDefaultConfig()
        self._storeConfig(defaultConfig, configPath)
        return defaultConfig
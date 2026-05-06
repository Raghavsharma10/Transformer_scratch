def _storeConfig(self, config, configPath):
        """ 
        Writes the config to the configPath.
        :param config a dict of config.
        :param configPath the path to the file to write to, intermediate dirs will be created as necessary.
        """
        self.logger.info("Writing to " + str(configPath))
        os.makedirs(os.path.dirname(configPath), exist_ok=True)
        with (open(configPath, 'w')) as yml:
            yaml.dump(config, yml, default_flow_style=False)
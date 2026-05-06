def get_registered_configs(self, instances=None):
        """ Return the persisted values of all config files registered with the config manager.
        """
        configs = self.state.get('config_files', {})
        if instances is not None:
            for config_file, config in configs.items():
                if config['instance_name'] not in instances:
                    configs.pop(config_file)
        return configs
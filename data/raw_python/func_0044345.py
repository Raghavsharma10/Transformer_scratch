def register_config_changes(self, configs, meta_changes):
        """ Persist config changes to the JSON state file. When a config
        changes, a process manager may perform certain actions based on these
        changes. This method can be called once the actions are complete.
        """
        for config_file in meta_changes['remove_configs'].keys():
            self._purge_config_file(config_file)
        for config_file, config in configs.items():
            if 'update_attribs' in config:
                config['attribs'] = config.pop('update_attribs')
            if 'update_instance_name' in config:
                config['instance_name'] = config.pop('update_instance_name')
            if 'update_services' in config or 'remove_services' in config:
                remove = config.pop('remove_services', [])
                services = config.pop('update_services', [])
                # need to prevent old service defs from overwriting new ones
                for service in config['services']:
                    if service not in remove and service not in services:
                        services.append(service)
                config['services'] = services
            self._register_config_file(config_file, config)
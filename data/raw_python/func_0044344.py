def determine_config_changes(self):
        """ The magic: Determine what has changed since the last time.

        Caller should pass the returned config to register_config_changes to persist.
        """
        # 'update' here is synonymous with 'add or update'
        instances = set()
        new_configs = {}
        meta_changes = { 'changed_instances' : set(),
                         'remove_instances' : [],
                         'remove_configs' : self.get_remove_configs() }
        for config_file, stored_config in self.get_registered_configs().items():
            new_config = stored_config
            try:
                ini_config = ConfigManager.get_ini_config(config_file, defaults=stored_config.defaults)
            except (OSError, IOError) as exc:
                log.warning('Unable to read %s (hint: use `rename` or `remove` to fix): %s', config_file, exc)
                new_configs[config_file] = stored_config
                instances.add(stored_config['instance_name'])
                continue
            if ini_config['instance_name'] is not None:
                # instance name is explicitly set in the config
                instance_name = ini_config['instance_name']
                if ini_config['instance_name'] != stored_config['instance_name']:
                    # instance name has changed
                    # (removal of old instance will happen later if no other config references it)
                    new_config['update_instance_name'] = instance_name
                meta_changes['changed_instances'].add(instance_name)
            else:
                # instance name is dynamically generated
                instance_name = stored_config['instance_name']
            if ini_config['attribs'] != stored_config['attribs']:
                # Ensure that dynamically generated virtualenv is not lost
                if ini_config['attribs']['virtualenv'] is None:
                    ini_config['attribs']['virtualenv'] = stored_config['attribs']['virtualenv']
                # Recheck to see if dynamic virtualenv was the only change.
                if ini_config['attribs'] != stored_config['attribs']:
                    self.create_virtualenv(ini_config['attribs']['virtualenv'])
                    new_config['update_attribs'] = ini_config['attribs']
                    meta_changes['changed_instances'].add(instance_name)
            # make sure this instance isn't removed
            instances.add(instance_name)
            services = []
            for service in ini_config['services']:
                if service not in stored_config['services']:
                    # instance has a new service
                    if 'update_services' not in new_config:
                        new_config['update_services'] = []
                    new_config['update_services'].append(service)
                    meta_changes['changed_instances'].add(instance_name)
                # make sure this service isn't removed
                services.append(service)
            for service in stored_config['services']:
                if service not in services:
                    if 'remove_services' not in new_config:
                        new_config['remove_services'] = []
                    new_config['remove_services'].append(service)
                    meta_changes['changed_instances'].add(instance_name)
            new_configs[config_file] = new_config
        # once finished processing all configs, find any instances which have been deleted
        for instance_name in self.get_registered_instances(include_removed=True):
            if instance_name not in instances:
                meta_changes['remove_instances'].append(instance_name)
        return new_configs, meta_changes
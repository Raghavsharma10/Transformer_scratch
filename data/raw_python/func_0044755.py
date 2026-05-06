def update(self):
        """ Add newly defined servers, remove any that are no longer present
        """
        configs, meta_changes = self.config_manager.determine_config_changes()
        self._process_config_changes(configs, meta_changes)
        self.supervisorctl('update')